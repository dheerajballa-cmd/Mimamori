<div align="center">

# 🛡️ Mimamori (見守り)
### *Proactive Multi-Modal Fall-Risk Forecasting & Causal Attribution for Elderly Care*

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Test Suite](https://img.shields.io/badge/pytest-49%20passed-success.svg)](tests/)
[![Architecture](https://img.shields.io/badge/design-causal--predictive-orange.svg)](ARCHITECTURE.md)
[![License](https://img.shields.io/badge/license-Research%20Use-lightgrey.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Week%203%20%7C%20Epic%205%20Complete-brightgreen.svg)](tasks.md)

<p align="center">
  <b>Moving eldercare from reactive post-fall detection to calibrated multi-day risk forecasting and actionable causal intervention.</b>
</p>

---

</div>

## 📌 Executive Summary

Traditional elderly fall solutions operate **reactively**—detecting an impact after trauma has already occurred. **Mimamori** is a predictive healthcare intelligence system engineered to forecast gradual shifts in fall vulnerability across multi-day and multi-week horizons. 

By unifying continuous biomechanical telemetry (wearable IMU), low-resolution privacy-preserving thermal vision (8×8 Grid-Eye), and non-invasive acoustic event tagging into standardized longitudinal health trajectories, Mimamori enables:
1. **Multi-Horizon Risk Forecasting**: Predicting individual fall probability shifts before incidents manifest.
2. **Causal Root-Cause Attribution**: Isolating structural drivers of elevated risk (e.g., gait degradation vs. sleep disruption vs. routine variance) using structural causal models (SCMs).
3. **Calibrated Actionable Interventions**: Generating constrained, human-in-the-loop recommendations for caregivers with built-in uncertainty quantification.
4. **Privacy by Design**: Ensuring zero raw RGB video or acoustic audio waveforms leave the edge boundary.

---

## 🏗️ End-to-End System Architecture

```
                                  ┌─────────────────────────────────────────────────────────┐
                                  │                Multi-Modal Sensor Inputs                │
                                  └────────────────────────────┬────────────────────────────┘
                                                               │
                 ┌─────────────────────────────────────────────┼─────────────────────────────────────────────┐
                 ▼                                             ▼                                             ▼
  ┌──────────────────────────────┐              ┌──────────────────────────────┐              ┌──────────────────────────────┐
  │     Wearable 9-Axis IMU      │              │      8×8 Thermal Sensor      │              │    Ambient Acoustic Stream   │
  │     (SisFall / KFall)        │              │     (eHome / TF-66 Proxy)    │              │       (Sounds of Home)       │
  └──────────────┬───────────────┘              └──────────────┬───────────────┘              └──────────────┬───────────────┘
                 │ Raw 100-200 Hz Tri-axial                    │ 8×8 Temperature Arrays                      │ Pre-computed Frame Classes
                 ▼                                             ▼                                             ▼
  ┌──────────────────────────────┐              ┌──────────────────────────────┐              ┌──────────────────────────────┐
  │     preprocess/gait.py       │              │    preprocess/thermal.py     │              │     preprocess/audio.py      │
  │  • 100 Hz Linear Resampling  │              │  • EMA Background Subtraction│              │  • 77→5 Ontology Mapping     │
  │  • 5.0s Windows (50% Step)   │              │  • Hot-Spot Silhouette BBox  │              │  • Windowed Statistical Agg  │
  │  • Cadence, Stride Var, Sway │              │  • Posture: Stand/Sit/Lie    │              │  • Zero Raw Audio Assertion  │
  └──────────────┬───────────────┘              └──────────────┬───────────────┘              └──────────────┬───────────────┘
                 │                                             │                                             │
                 │ Extracted Biomechanical Features            │ Posture & Delta Spikes                      │ Acoustic Event Distributions
                 └─────────────────────────────────────────────┼─────────────────────────────────────────────┘
                                                               │
                                                               ▼
                                       ┌───────────────────────────────────────────────┐
                                       │    Epic 6: Daily Multi-Modal Aggregator       │
                                       │           (preprocess/aggregate.py)           │
                                       │        Schema: DailyFeatureVector (24h)       │
                                       └───────────────────────┬───────────────────────┘
                                                               │
                                                               ▼
                                       ┌───────────────────────────────────────────────┐
                                       │     Longitudinal Sequence Pretraining         │
                                       │          (NHATS Panel / Transformer)          │
                                       └───────────────────────┬───────────────────────┘
                                                               │
                                                               ▼
                                       ┌───────────────────────────────────────────────┐
                                       │      Causal Structure & Attribution Layer     │
                                       │      (DoWhy / Causal-Learn / Top-k Causes)    │
                                       └───────────────────────┬───────────────────────┘
                                                               │
                                                               ▼
                                       ┌───────────────────────────────────────────────┐
                                       │     Caregiver Dashboard & Feedback Loop       │
                                       │    (Streamlit / Structured LLM Guidances)     │
                                       └───────────────────────────────────────────────┘
```

---

## 📊 Dataset Ingestion & Modality Matrix

The pipeline orchestrates heterogeneous public and synthetic benchmark datasets through unified programmatic loaders ([`data/loaders/loaders.py`](file:///D:/Mimamori/data/loaders/loaders.py)):

| Dataset | Modality / Sensor | Scope / Demographics | Role in Mimamori |
|---|---|---|---|
| **SisFall** | 9-axis Waist IMU (200 Hz) | 38 subjects (young + older adults, 19 ADLs, 15 falls) | Gait dynamics & baseline biomechanical extraction |
| **KFall & Elderly** | Lumbar IMU (100 Hz) | 32 young + 10 older adults, continuous trials | Cross-cohort gait generalization & proxy label validation |
| **NHATS Panel** | Survey & 7-Day Triaxial Actigraphy | 8,000+ nationally representative US Medicare beneficiaries | Primary longitudinal representation pretraining |
| **eHomeSeniors** | Melexis 8×8 Grid-Eye + Omron PIR | Continuous smart-home elderly living apartments | Ambient thermal occupancy & multi-day routine tracking |
| **TF-66** | Thermal Video Annotations | 66 volunteers, 812 annotated fall/ADL trials | Fall event label taxonomy & duration calibration |
| **Sounds of Home** | Ambient Acoustic Inferences | 14 domestic home deployments (multi-week continuous) | Passive acoustic event classification & routine shifts |
| **Synthetic Longitudinal v0.1** | Structural Causal Model Trajectories | Multi-week counterfactual participant trajectories | Ground-truth causal attribution & ablation validation |

---

## 🔬 Core Engineering & Preprocessing Modules

### 1. Zero-Leakage Split Protocol (`data/splits.py`)
To prevent severe metric inflation caused by intra-subject gait/posture correlation across windows:
* **Subject-Level Partitioning**: All dataset partitions (Train: 70%, Val: 15%, Test: 15%) are computed strictly over unique **Subject / Participant IDs**.
* **Determinism & Reproducibility**: Splits are pre-computed, verified, and frozen via fixed pseudo-random seeding.

### 2. Biomechanical Gait Extraction (`preprocess/gait.py`)
* **Standardized Rate**: Interpolates variable IMU signals to a standard $f_s = 100\text{ Hz}$.
* **Temporal Windowing**: 5.0-second sliding windows with 50% overlap ($N = 500$ samples/window).
* **Feature Signatures**:
  * **Cadence ($C$)**: Step rate computed via 4th-order low-pass Butterworth filter ($f_c = 3.0\text{ Hz}$) and adaptive refractory peak detection on acceleration norm $\|\mathbf{a}(t)\|$.
  * **Stride Time Variability ($\sigma_{\Delta t}$)**: Standard deviation of consecutive stride intervals $\Delta t_i = t_{i+1} - t_i$.
  * **Sway Index ($S_{\text{gyr}}$)**: Root-mean-square (RMS) magnitude of angular velocity $\sqrt{\frac{1}{N}\sum \|\boldsymbol{\omega}(t)\|^2}$.

### 3. Thermal Posture & Fall Feature Extraction (`preprocess/thermal.py`)
* **Background Modeling**: Exponential Moving Average (EMA) thermal baseline model ($\alpha = 0.05$):
  $$\mathbf{B}_t = (1 - \alpha)\mathbf{B}_{t-1} + \alpha \mathbf{F}_t$$
* **Foreground Silhouette & Centroid**: Isolates person hot-spots exceeding background temperature offset ($\Delta T > 2.0^\circ\text{C}$). Computes center of mass and bounding box dimensions $(h, w)$.
* **Posture Classification**:
  * **Standing**: Aspect Ratio $\frac{h}{w} > 1.4$
  * **Sitting**: $0.7 < \frac{h}{w} \le 1.4$
  * **Lying**: $\frac{h}{w} \le 0.7$
* **Rapid Motion Spike**: Computes frame-to-frame $L_2$ norm delta $\|\mathbf{F}_t - \mathbf{F}_{t-1}\|_2$ for sudden descent detection.

### 4. Acoustic Event Classification (`preprocess/audio.py`)
* **Ontology Mapping**: Maps 77 raw acoustic prediction classes into 5 normalized high-level categories:
  * `0: silence` (ambient background, white noise, static)
  * `1: routine` (appliance hums, television, clock ticking, kitchen sounds)
  * `2: footsteps` (shuffling, door closures, walking movement)
  * `3: distress` (labored breathing, coughing fits, snoring, audible alarms)
  * `4: fall` (sudden impacts, loud thuds, transient shock vibrations)
* **Privacy Assurance**: Programmatically asserts zero unencrypted raw audio streams (`.wav`, `.mp3`) exist in the storage boundary.

---

## 📈 Implementation Roadmap & Status

```
[████████████████████░░░░░░░░░░░░░░░░░░░░] 35% Completed (Epics 1–5 / 16)
```

- [x] **Week 1: Foundation**
  - [x] Epic 1: Repository scaffolding, pinned environment setup, reproducibility checks.
  - [x] Epic 2: Data acquisition, unified programmatic loaders, frozen subject splits.
- [x] **Weeks 2–3: Multi-Modal Feature Extraction**
  - [x] Epic 3: IMU gait extraction pipeline (`gait.py`), cross-dataset validation on SisFall & KFall.
  - [x] Epic 4: Thermal frame parsing, posture categorization, RF fall classifier (`thermal.py`).
  - [x] Epic 5: Acoustic event extraction, privacy verification, balanced event classifier (`audio.py`).
- [ ] **Weeks 3–4: Aggregation & Representation Pretraining**
  - [ ] Epic 6: Unified `DailyFeatureVector` longitudinal aggregator (`aggregate.py`).
  - [ ] Epic 7: Self-supervised longitudinal pretraining on NHATS actigraphy corpus.
- [ ] **Weeks 5–6: Risk Forecasting Layer**
  - [ ] Epic 8: Transformer multi-horizon fall risk forecasting head & proxy evaluation.
- [ ] **Weeks 7–9: Causal Attribution Layer**
  - [ ] Epic 9: Synthetic longitudinal causal graph dataset & SCM specification.
  - [ ] Epic 10: Structural causal discovery (DoWhy / PC algorithm) vs. attention saliency baselines.
- [ ] **Weeks 10–11: Interventions & Feedback**
  - [ ] Epic 11: Schema-constrained LLM caregiver recommendation generation.
  - [ ] Epic 12: Caregiver confirmation logging, online recalibration & personalization drift.
- [ ] **Weeks 12–14: Evaluation, Dashboard & Writeup**
  - [ ] Epic 13: 4-way ablation harness (Pretraining / Causal / Feedback / Multi-horizon).
  - [ ] Epic 14: Interactive Streamlit caregiver dashboard (`dashboard/app.py`).
  - [ ] Epic 16: Research paper writeup & benchmark disclosure.

---

## 🚀 Quickstart & Verification

### 1. Environment Setup

```powershell
# Clone repository
git clone https://github.com/your-org/mimamori.git
cd mimamori

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Execute Complete Test Suite
Run all 49 unit tests across gait, thermal, audio, and split modules:
```powershell
pytest tests/ -v
```

### 3. Run Pipeline Verification Scripts
Execute modular verification runners using real and benchmark data:
```powershell
# Verify IMU Gait Feature Extraction (SisFall + KFall)
python scripts/run_epic3.py

# Verify Thermal Silhouette & Posture Extraction (eHomeSeniors + TF-66)
python scripts/run_epic4.py

# Verify Ambient Audio Event Classification & Privacy Check (Sounds of Home)
python scripts/run_epic5.py
```

---

## 🔒 Privacy, Safety & Ethical Commitments

1. **Non-Invasive Sensing**: Mimamori explicitly rejects invasive optical RGB surveillance. All optical spatial data is strictly constrained to low-resolution (8×8) heat matrices incapable of facial re-identification.
2. **On-Device Feature Abstraction**: High-frequency acoustic signals and raw telemetry are transformed into statistical aggregates and categorical probabilities at the ingress boundary.
3. **Calibrated Caution**: Interventions and risk scores are delivered with rigorous uncertainty metrics and require human caregiver validation before triggering clinical escalations.

---

## 📄 License & Attribution

This repository is maintained for research and academic validation under the Mimamori Research License. Built with reference to the National Health and Aging Trends Study (NHATS), SisFall, KFall, and eHomeSeniors benchmark consortia.
