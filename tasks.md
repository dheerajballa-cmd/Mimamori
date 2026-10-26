# Implementation Plan: Mimamori (14-Week Flagship Build)

## Overview

This plan builds Mimamori from repo scaffold to an ablation-backed, demoable prototype using only public + self-built synthetic data — no dependency on a real household pilot to reach a defensible result.

**Timeline**: ~14 weeks (part-time, alongside coursework)
**Implementation Language**: Python 3.11
**Philosophy**: Every stage produces an artifact the next stage consumes; nothing downstream is blocked on a real pilot landing.

---

## Week 1: Foundation

### Epic 1: Repo Scaffold & Environment

- [x] 1.1 Create repo structure (`data/`, `pretrain/`, `forecast/`, `causal/`, `intervene/`, `feedback/`, `eval/`, `edge/`, `dashboard/`, `configs/`)
- [x] 1.2 Set up environment: PyTorch, Transformers, PyTorch Lightning, DoWhy, causal-learn, scikit-learn, onnx/onnxruntime, netcal, outlines, streamlit
- [x] 1.3 Initialize `wandb` (or `mlflow`) project — required before any training run, not added later
- [x] 1.4 Checkpoint — Verify environment reproducibility (`pip freeze` pinned, fresh-venv smoke test)

### Epic 2: Dataset Acquisition & Access

- [x] 2.1 Acquire and license-check **SisFall**, **KFall**, **KFall elderly add-in** (IMU)
- [x] 2.2 Acquire and license-check **NHATS** + **NHATS accelerometry** — confirm public-use vs. restricted-access files; file a data use agreement if the accelerometry sub-study requires it
- [x] 2.3 Acquire **eHomeSeniors**, **Sounds of Home**, **Thermal Fall 66**, **CIRDO**
- [x] 2.4 Write a per-dataset `data/README_<dataset>.md` documenting license, modality, subject count/demographics, and known limitations (age skew, lab-only sessions, etc.)
- [x] 2.5 Define and document a **subject-level split protocol** (`data/splits.py`) applied uniformly across SisFall, KFall, KFall elderly add-in, Thermal Fall 66, CIRDO, and Sounds of Home — splitting by window/session instead of by subject leaks a subject's gait/posture/voice signature across train and test and silently inflates every downstream metric. This protocol is what Epics 3, 4, 5, 8, and 10 must all call, not re-implement.
- [x] 2.6 Checkpoint — All datasets loadable via a common `data/loaders/` interface returning raw samples, with subject-level train/val/test assignments pre-computed and frozen before any feature extraction begins

---

## Week 2: Preprocessing Pipelines

### Epic 3: Gait/IMU Feature Extraction

- [x] 3.1 Build `preprocess/gait.py` — resample to 100Hz, 5–10s windows, 50% overlap
- [x] 3.2 Extract cadence, stride variability, sway index from **SisFall**
- [x] 3.3 Validate feature extractor generalizes to **KFall** and **KFall elderly add-in** (cross-dataset check), using the subject-level splits from 2.5
- [x] 3.4 Write unit tests for windowing and feature extraction edge cases (short sessions, sensor dropout)

### Epic 4: Thermal Posture/Fall Extraction

- [x] 4.1 Build `preprocess/thermal.py` — silhouette/posture classification from low-res thermal frames
- [x] 4.2 Train on **Thermal Fall 66**
- [x] 4.3 Cross-validate on **CIRDO**; log generalization gap explicitly (this is a named risk in ARCHITECTURE.md §9)

### Epic 5: Ambient Audio Event Classification

- [x] 5.1 Build `preprocess/audio.py` — event classifier (routine, footsteps, distress, fall sounds)
- [x] 5.2 Train/validate on **Sounds of Home**
- [x] 5.3 Confirm raw audio is discarded post-extraction; only event labels persist (privacy requirement, ARCHITECTURE.md §13)

### Epic 6: Daily/Weekly Feature Aggregation

- [x] 6.1 Define shared `DailyFeatureVector` schema (per ARCHITECTURE.md §3)
- [x] 6.2 Build `preprocess/aggregate.py` — merge gait + thermal + audio outputs into daily vectors
- [x] 6.3 Adapt **eHomeSeniors** multi-day sensor logs into the same schema
- [x] 6.4 Adapt **NHATS** + **NHATS accelerometry** longitudinal records into the same schema
- [x] 6.5 Derive `sleep_proxy` from NHATS self-reported sleep items + audio-inferred nighttime activity (**Sounds of Home**) — no dataset measures sleep directly, so document this as a lower-confidence feature (ARCHITECTURE.md §9)
- [x] 6.6 Checkpoint — One unified daily-feature dataset spanning all sources, schema-validated

### Epic 6b: Outputs & Verification (Stage 1 — Preprocessing)

- [x] 6b.1 Save datasets to `/data/processed/` with exact filenames:
  - `daily_features_nhats.pkl` (NHATS + accelerometry aggregated)
  - `daily_features_ehome.pkl` (eHomeSeniors in same schema)
  - `sessions_sisfall_kfall.pkl` (SisFall/KFall synthetic weeks)
  - `splits.json` (subject-level train/val/test assignments, frozen before feature extraction)
  - `feature_schema.json` (DailyFeatureVector spec)
- [x] 6b.2 Create `data/verify_preprocessing.py` that checks:
  - File existence and sizes
  - dtypes and shapes match schema
  - NaN counts (must be <5%)
  - t-SNE plot colored by dataset source (should show longitudinal vs. lab-trial clusters)
- [x] 6b.3 Run `python data/verify_preprocessing.py`; if any check fails, STOP and debug
- [x] 6b.4 Log to wandb: `feature_distribution_plots.png`, `nan_counts.json`, `sequence_length_stats.json`
- [x] 6b.5 Create and commit `data/PREPROCESSING_REPORT.md` documenting:
  - Total sequences per dataset
  - Sequence lengths and shapes
  - Any missing values and how they were handled
  - Cross-dataset generalization gaps (Thermal Fall 66 vs. CIRDO)
- [x] 6b.6 Checkpoint — All files exist, verification passes, report is complete before proceeding to pretraining

---

## Week 3–4: Self-Supervised Pretraining

### Epic 7: Masked Sequence Pretraining

- [x] 7.1 Implement PatchTST-style encoder (`pretrain/model.py`) via `transformers.PatchTSTModel`
- [x] 7.2 Implement masked-patch reconstruction objective (30–40% masking) in `pretrain/train.py`
- [x] 7.3 Pretrain primarily on **NHATS** + **NHATS accelerometry**; use **eHomeSeniors** as secondary corpus
- [x] 7.4 Log training curves and reconstruction loss to `wandb`
- [x] 7.5 Checkpoint — Sanity check: cluster embeddings by NHATS functional-status labels (unsupervised); verify separation before proceeding (ARCHITECTURE.md §5). **Do not proceed to Week 5 until this passes.**

### Epic 7b: Outputs & Verification (Stage 2 — Pretraining)

- [x] 7b.1 Save encoder to `/pretrain/checkpoints/encoder_best.pt`
- [x] 7b.2 Save training metadata to `/pretrain/checkpoints/pretraining_metadata.json` containing:
  - Final reconstruction loss value
  - Number of training epochs
  - Learning rate, batch size, masking ratio
  - Dataset sizes (NHATS, eHomeSeniors)
- [x] 7b.3 Create `scripts/verify_stage_2.py` that:
  - Loads encoder and computes embeddings on NHATS test set
  - Computes silhouette score against functional-status labels
  - Asserts `silhouette_score > 0.3`
  - Generates t-SNE/UMAP plot colored by status and saves to `/pretrain/checkpoints/embeddings_by_status.png`
- [x] 7b.4 Run `python scripts/verify_stage_2.py`; if silhouette < 0.3, STOP and debug pretraining
- [x] 7b.5 Log to wandb artifacts:
  - Encoder weights (`encoder_best.pt`)
  - `pretraining_loss_curve.png` (loss over epochs)
  - `embeddings_by_status.png` (t-SNE/UMAP)
  - `pretraining_metadata.json`
- [x] 7b.6 Create and commit `pretrain/PRETRAINING_REPORT.md` documenting:
  - Final reconstruction loss
  - Silhouette score value
  - Embedding dimensionality
  - Whether NHATS subgroups separate (must be Yes)
- [x] 7b.7 Checkpoint — Encoder saved, silhouette > 0.3, report complete. Do NOT proceed to forecasting (Week 5) until this passes.

---

## Week 5–6: Multi-Horizon Forecasting

### Epic 8: Forecasting Head Fine-Tuning

- [x] 8.1 Attach quantile-regression forecasting head (short: 3–7 day, long: 4–8 week) to pretrained encoder
- [x] 8.2 **Resolve the granularity mismatch before writing any fine-tuning code**: the Epic 7 encoder expects daily/weekly aggregated sequences, but SisFall/KFall/KFall-elderly are single-session lab trials with no multi-day structure. Adopt the explicit workaround — concatenate each subject's trial windows in sequence and treat the result as a short synthetic "week" — and implement it as a documented, named transform (`forecast/session_to_sequence.py`), not an ad hoc reshape buried in the training script. State this workaround plainly in the writeup (§16.2) as a known limitation, not an implementation detail.
- [x] 8.3 Construct proxy supervised labels from **SisFall**/**KFall**/**KFall elderly add-in** (fall-vs-ADL, gait-severity), applied to the synthetic-week sequences from 8.2, using the subject-level splits from 2.5
- [x] 8.4 Fine-tune with PyTorch Lightning; checkpoint best model by validation AUPRC
- [x] 8.5 Implement baseline: logistic regression / XGBoost on same-day features, no pretraining, no temporal structure
- [x] 8.6 Implement confidence-gated abstention — below a calibrated confidence threshold, output "not confident enough to flag" instead of a forced risk score (this was named as a safety principle in the flagship doc but had no build task until now)
- [x] 8.7 Export a `risk_delta` (change in forecasted risk between consecutive forecasting windows) as a first-class output of the forecasting head — this is the exact quantity Epic 10.3's causal ranking consumes, and currently no task produces it explicitly
- [x] 8.8 Checkpoint — Fine-tuned model must beat the static baseline on the proxy task; debug here before proceeding, not later

### Epic 8b: Outputs & Verification (Stage 3 — Forecasting)

- [x] 8b.1 Save models to `/forecast/checkpoints/`:
  - `forecaster_best.pt` (encoder + forecasting head)
  - `baseline_xgboost.pkl` (static baseline for comparison)
  - `forecast_metadata.json` (architecture, hyperparams, training time)
- [x] 8b.2 Generate and save evaluation plots:
  - `calibration_curve_short_horizon.png` (reliability diagram for short-horizon forecast)
  - `calibration_curve_long_horizon.png` (reliability diagram for long-horizon forecast)
  - `auroc_comparison.png` (bar chart: forecaster vs. baseline AUROC per horizon)
- [x] 8b.3 Save metrics to `/forecast/checkpoints/forecast_metrics.json`:
  ```json
  {
    "auroc_short": X,
    "auroc_long": Y,
    "auprc_short": Z,
    "auprc_long": W,
    "calibration_error_short": ECE_short,
    "calibration_error_long": ECE_long,
    "abstention_rate": A,
    "lead_time_to_event_median": L
  }
  ```
- [x] 8b.4 Create `scripts/verify_stage_3.py` that:
  - Loads forecaster and baseline
  - Computes AUROC for both on validation set
  - Asserts `forecaster_auroc > baseline_auroc`
  - Asserts `calibration_error < 0.15` (well-calibrated)
  - Prints all metrics
- [x] 8b.5 Run `python scripts/verify_stage_3.py`; if forecaster AUROC <= baseline AUROC, STOP and debug gait feature extraction or model architecture
- [x] 8b.6 Log to wandb artifacts: `forecaster_best.pt`, `baseline_xgboost.pkl`, all PNG plots, `forecast_metrics.json`
- [x] 8b.7 Create and commit `forecast/FORECASTING_REPORT.md` documenting:
  - AUROC/AUPRC for forecaster and baseline per horizon
  - Calibration errors
  - Abstention rate
  - Session-to-sequence workaround and why it was necessary
- [x] 8b.8 Checkpoint — Forecaster beats baseline, calibration acceptable, report complete. Do NOT proceed to causal layer until this passes.

---

## Week 7–9: Causal Attribution Layer

### Epic 9: Synthetic Longitudinal Causal Dataset

- [ ] 9.1 Design structural causal model (SCM): nodes = `sleep_quality`, `routine_deviation`, `gait_stability`, `medication_adherence_proxy`, `fall_risk`; specify edges and noise model
- [ ] 9.2 Implement `causal/simulate.py` — generate multi-week synthetic household trajectories from the SCM
- [ ] 9.3 Generate train/val/test splits with known ground-truth causal graph retained for evaluation
- [ ] 9.4 Document the SCM design choices in `causal/README.md` (this is the project's most scrutinized component — document it well)

### Epic 10: Causal Structure Learning & Attribution

- [ ] 10.1 Integrate `DoWhy` for intervention-style effect estimation
- [ ] 10.2 Integrate `causal-learn` (PC algorithm / Granger causality) as a structure-learning cross-check against the known synthetic graph
- [ ] 10.3 Implement top-k root-cause ranking given a forecasted `risk_delta` (from Epic 8.7 — the exact interface between the forecasting and causal layers)
- [ ] 10.4 Evaluate top-k root-cause accuracy against synthetic ground truth
- [ ] 10.5 Implement attention-weight saliency baseline (correlational) for Ablation #2 comparison
- [ ] 10.6 Apply attribution method to real **eHomeSeniors**/**NHATS** trajectories as a plausibility check (explicitly labeled as non-validating in any writeup)
- [ ] 10.7 Checkpoint — Causal method must beat saliency baseline on synthetic top-k accuracy

### Epic 10b: Outputs & Verification (Stage 4 — Causal)

- [ ] 10b.1 Save causal artifacts to `/causal/checkpoints/`:
  - `scm_ground_truth.json` (hand-specified edges, noise model, node definitions)
  - `synthetic_trajectories_train.pkl`, `_val.pkl`, `_test.pkl` (generated from SCM)
  - `causal_model_best.pt` (DoWhy/causal-learn fitted model weights)
  - `causal_metadata.json` (method used, hyperparams, training time)
- [ ] 10b.2 Save comparison baselines:
  - `saliency_baseline.pkl` (attention-weight saliency method for Ablation #2)
- [ ] 10b.3 Create `scripts/verify_stage_4.py` that:
  - Loads causal model and saliency baseline
  - Evaluates both on synthetic test set
  - Computes top-1 and top-3 root-cause accuracy for each
  - Asserts `causal_top1 > saliency_top1` AND `causal_top3 > saliency_top3`
  - Prints metrics
- [ ] 10b.4 Run `python scripts/verify_stage_4.py`; if causal doesn't beat saliency on top-k accuracy, STOP and revisit SCM or attribution method
- [ ] 10b.5 Generate and save evaluation plots:
  - `top_k_accuracy_comparison.png` (bar chart: causal vs. saliency for top-1, top-3, top-5)
  - `causal_attribution_examples.png` (3–5 example trajectories with ranked attributed factors)
- [ ] 10b.6 Save metrics to `/causal/checkpoints/causal_metrics.json`:
  ```json
  {
    "top1_accuracy_causal": X,
    "top1_accuracy_saliency": Y,
    "top3_accuracy_causal": Z,
    "top3_accuracy_saliency": W,
    "mean_reciprocal_rank_causal": MRR_c,
    "mean_reciprocal_rank_saliency": MRR_s
  }
  ```
- [ ] 10b.7 Log to wandb artifacts: `scm_ground_truth.json`, `causal_model_best.pt`, all PNG plots, `causal_metrics.json`
- [ ] 10b.8 Create and commit `causal/CAUSAL_README.md` documenting:
  - SCM design choices (why those specific edges, noise model rationale)
  - Top-k accuracy results (causal vs. saliency)
  - Whether real-data plausibility check passed (not a validation, only a sanity check)
- [ ] 10b.9 Checkpoint — Causal beats saliency, metrics saved, report complete. Do NOT proceed to feedback loop until this passes.

---

## Week 10: Intervention Generation

### Epic 11: LLM Intervention Generator

- [ ] 11.1 Select and serve local instruction-tuned LLM (Llama/Qwen-family 7–8B) via `ollama` or `transformers`
- [ ] 11.2 Design structured prompt (ranked factors + confidence + household context → JSON output)
- [ ] 11.3 Enforce JSON schema via `outlines`/`guidance` grammar constraints
- [ ] 11.4 Run small human-rated evaluation (10–15 examples) for actionability/clarity

### Epic 11b: Outputs & Verification (Stage 5 — LLM)

- [ ] 11b.1 Save LLM config and outputs to `/intervene/`:
  - `llm_config.yaml` (model name, temperature, prompt template, grammar constraints)
  - `generated_interventions_test_set.jsonl` (100+ examples of ranked factors → JSON recommendations)
- [ ] 11b.2 Create `scripts/verify_stage_5.py` that:
  - Loads all generated interventions from JSONL
  - Attempts to parse each as JSON
  - Checks all required keys exist (summary, recommended_action, urgency, confidence)
  - Asserts 100% parse success rate (no malformed JSON)
  - Counts parse failures
- [ ] 11b.3 Run `python scripts/verify_stage_5.py`; if parse failures > 0, STOP and adjust prompt/grammar constraints
- [ ] 11b.4 Save human ratings to `/intervene/human_ratings.csv`:
  - Columns: example_id, actionability (1-5), clarity (1-5), rater_notes
  - 10–15 rated examples minimum
  - Compute mean actionability and clarity scores
- [ ] 11b.5 Create `scripts/compute_llm_metrics.py` that:
  - Loads ratings
  - Computes mean actionability, mean clarity
  - Asserts mean_actionability >= 3.0 (acceptable threshold)
- [ ] 11b.6 Run `python scripts/compute_llm_metrics.py`; if mean actionability < 3.0, STOP and revise prompt
- [ ] 11b.7 Log to wandb: `llm_config.yaml`, `generated_interventions_test_set.jsonl`, `human_ratings.csv`, `llm_metrics.json`
- [ ] 11b.8 Create and commit `intervene/LLM_REPORT.md` documenting:
  - Model used and configuration
  - Parse success rate
  - Mean actionability and clarity scores
- [ ] 11b.9 Checkpoint — JSON parse 100%, actionability >= 3.0. Proceed to feedback loop.

---

## Week 11: Caregiver Feedback Loop

### Epic 12: Confirmation & Online Recalibration

- [ ] 12.1 Implement `feedback/log.py` — structured feedback event logging (confirm/override/dismiss)
- [ ] 12.2 Implement online edge-weight recalibration of the causal graph from feedback events
- [ ] 12.3 Add automatic revert if attribution accuracy on the synthetic held-out set degrades beyond a threshold (guards against caregiver-bias drift, ARCHITECTURE.md §9)
- [ ] 12.4 Checkpoint — Simulate multiple feedback cycles on synthetic data; verify attribution accuracy improves (not flat/degrading) — this is Ablation #3's evidence

### Epic 12b: Outputs & Verification (Stage 6 — Feedback Loop)

- [ ] 12b.1 Save feedback artifacts to `/feedback/`:
  - `feedback_log.jsonl` (structured log of caregiver confirmations/overrides over 20+ cycles)
  - `causal_graph_evolution.json` (edge weights at cycle 0, 5, 10, 15, 20)
- [ ] 12b.2 Create `scripts/verify_stage_6.py` that:
  - Loads feedback log and causal graph evolution
  - Plots attribution accuracy vs. cycle number
  - Checks for trend: should increase or at least not degrade significantly
  - Asserts `accuracy_at_cycle_20 >= accuracy_at_cycle_0` (no degradation)
  - Generates `attribution_accuracy_over_cycles.png`
- [ ] 12b.3 Run `python scripts/verify_stage_6.py`; if accuracy degrades, STOP and check recalibration logic or automatic revert threshold
- [ ] 12b.4 Save metrics to `/feedback/feedback_metrics.json`:
  ```json
  {
    "initial_attribution_accuracy": X,
    "final_attribution_accuracy": Y,
    "accuracy_improvement": Y - X,
    "num_cycles": 20,
    "num_confirms": C,
    "num_overrides": O,
    "causal_graph_drift_score": D
  }
  ```
- [ ] 12b.5 Log to wandb: `feedback_log.jsonl`, `causal_graph_evolution.json`, `attribution_accuracy_over_cycles.png`, `feedback_metrics.json`
- [ ] 12b.6 Create and commit `feedback/FEEDBACK_LOOP_REPORT.md` documenting:
  - Attribution accuracy trend (should improve or stay flat, not degrade)
  - How many caregiver confirms vs. overrides
  - Any causal graph edge reversals or weight collapses
- [ ] 12b.7 Checkpoint — Attribution accuracy doesn't degrade, metrics logged. Proceed to personalization and ablations.

### Epic 12c: Per-Household Personalization

- [ ] 12c.0 Define and document execution order relative to Epic 12: personalization (baseline drift correction) runs **before** causal recalibration in each cycle — the forecasting baseline is adjusted first so that what reaches the causal layer is a risk delta relative to the household's own norm, not the population norm. Without this ordering, Epic 12's causal recalibration risks absorbing a signal that 12c's baseline drift should have already accounted for, and the two modules would end up correcting for the same thing twice.
- [ ] 12c.1 Implement per-household baseline drift correction — the forecasting model's reference "normal" for gait/routine/sleep should shift with a household's own history rather than staying fixed to the population baseline it was pretrained/fine-tuned on
- [ ] 12c.2 Validate on synthetic multi-household trajectories: confirm the personalized baseline reduces false alerts for households with a stable-but-atypical routine (this was in the original architecture's pipeline diagram but had no implementation task until now)
- [ ] 12c.3 Checkpoint — Personalized vs. non-personalized forecasting compared on synthetic data; confirm the drift correction doesn't mask genuine risk changes (i.e., it adapts the baseline, not the alert threshold), and confirm the 12c.0 ordering doesn't produce double-counted corrections against Epic 12's recalibration

---

## Week 12: Evaluation & Ablations

### Epic 13: Ablation Harness

- [ ] 13.1 Build `eval/run_ablation.py` with config flags for all four ablations (no pretraining / no causal layer / no feedback loop / single-horizon)
- [ ] 13.2 Run and log all four ablations against the full system
- [ ] 13.3 Compute calibration curves via `netcal`, lead-time-to-event, abstention rate
- [ ] 13.4 Implement false-negative-weighted safety metric (missed real risk weighted more heavily than false positives) — not captured by plain AUROC/AUPRC, and explicitly called for in the flagship doc's safety design
- [ ] 13.5 Produce the final ablation comparison table (per ARCHITECTURE.md §10)
- [ ] 13.6 Checkpoint — All four ablations show the expected metric degradation; if any doesn't, investigate before writeup

### Epic 13b: Outputs & Verification (Stage 7 — Evaluation)

- [ ] 13b.1 Save ablation outputs to `/eval/`:
  - `ablation_01_no_pretraining/` (full logs + metrics from this run)
  - `ablation_02_no_causal_layer/` (full logs + metrics)
  - `ablation_03_no_feedback_loop/` (full logs + metrics)
  - `ablation_04_single_horizon/` (full logs + metrics)
- [ ] 13b.2 Build `ablation_results_table.csv` — 5 rows (full system + 4 ablations) × columns: AUROC, AUPRC, calibration error, lead-time, top-k causal accuracy, abstention rate, false-negative-weighted score
- [ ] 13b.3 Create `scripts/verify_stage_7.py` that:
  - Loads the ablation results table
  - For each ablation, checks the metric it's supposed to degrade actually dropped relative to the full system (e.g., no-pretraining → forecasting AUROC drop; no-causal → top-k accuracy drop; no-feedback → flat/declining attribution-accuracy trend; single-horizon → AUROC drop at the untrained horizon)
  - Flags (does not hard-fail) any ablation where the expected metric did NOT degrade, since this may be a genuine finding rather than a bug — but it must be investigated and explained, not silently accepted
- [ ] 13b.4 Run `python scripts/verify_stage_7.py`; document the outcome for every ablation, expected or not
- [ ] 13b.5 Generate `final_metrics_comparison.png` — bar chart, full system vs. each ablation, across all key metrics
- [ ] 13b.6 Log to wandb: `ablation_results_table.csv`, all four ablation directories, `final_metrics_comparison.png`
- [ ] 13b.7 Create and commit `eval/EVALUATION_REPORT.md` documenting which ablation caused the biggest drop in which metric, whether each result matched the expected hypothesis, and an explanation for any that didn't
- [ ] 13b.8 Checkpoint — All four ablations documented with explained outcomes, table and chart saved. This is the evidence base for the writeup (Epic 16) — do not start writing before this passes.

---

## Week 13: Dashboard & (Optional) Edge Deployment

### Epic 14: Caregiver Dashboard

- [ ] 14.1 Build Streamlit dashboard (`dashboard/app.py`) — risk trajectory, ranked factors, LLM recommendation, confirm/override control
- [ ] 14.2 Wire dashboard to feedback log (Epic 12)

### Epic 14b: Outputs & Verification (Stage 8 — Dashboard)

- [ ] 14b.1 Save dashboard artifacts to `/dashboard/`:
  - `app.py` (Streamlit code — keep under ~200 lines; this is deliberately not the project's technical contribution)
  - `demo_household_trajectory.json` (one synthetic multi-week household trajectory used for the demo)
- [ ] 14b.2 Create `scripts/verify_stage_8.py` (a manual-run checklist script, not a pure assertion test, since this stage is UI-driven) that walks through: load demo trajectory → risk alert renders → causal factors render → LLM recommendation renders → confirm/override button writes to `feedback/log.jsonl` → causal graph edge weights change as a result
- [ ] 14b.3 Run the dashboard end-to-end against the demo trajectory; capture `demo_screenshots.png` at each step (risk view, attribution view, recommendation, post-confirmation state)
- [ ] 14b.4 Confirm no unhandled exceptions occur during the full walkthrough
- [ ] 14b.5 Create and commit `dashboard/DASHBOARD_REPORT.md` documenting what the demo shows and confirming the full loop (risk → attribution → intervention → confirm → recalibrate) executes end-to-end
- [ ] 14b.6 Checkpoint — Full loop demoable without error, screenshots captured, report complete

### Epic 15: Edge Deployment (if real pilot pursued)

- [ ]* 15.1 Quantize pretrained encoder + forecasting head to INT8 via `onnxruntime`/`torch.quantization`
- [ ]* 15.2 Deploy on Raspberry Pi 4/5 with IMU + thermal sensor; verify local inference latency
- [ ]* 15.3 Confirm no raw sensor data leaves the device in the deployed pipeline (verify, don't assume)
- [ ]* 15.4 Recruit 2–5 consenting households; run short pilot

*(Starred tasks are optional — the project's core technical contribution does not depend on them; see ARCHITECTURE.md §9.)*

---

## Week 14: Writeup

### Epic 16: Research Writeup

- [ ] 16.1 Structure the paper/report around the four ablations, not a narrative
- [ ] 16.2 Document dataset limitations explicitly (age skew in SisFall/KFall base cohorts, lab-only fall sessions, synthetic-only causal validation, no direct sleep dataset)
- [ ] 16.3 Draft target-venue framing (gerontechnology / ubiquitous-computing workshop or conference — realistic first-submission target)
- [ ] 16.4 Final checkpoint — All ablation results, dataset documentation, and limitations sections complete before submission

---

## Notes

### Dataset-to-Stage Map

| Dataset | Used In |
|---|---|
| SisFall, KFall, KFall elderly add-in | Epic 3 (gait features), Epic 8 (proxy forecasting labels) |
| NHATS, NHATS accelerometry | Epic 6 (aggregation), Epic 7 (pretraining — primary corpus) |
| eHomeSeniors | Epic 6 (aggregation), Epic 10 (real-data plausibility check) |
| Sounds of Home | Epic 5 (audio events) |
| Thermal Fall 66 | Epic 4 (thermal training) |
| CIRDO | Epic 4 (thermal cross-validation) |
| Synthetic longitudinal causal dataset | Epic 9–10 (causal layer ground truth) |

### Checkpoints

Every epic ends with a checkpoint that must pass before the next epic starts — this mirrors the flagship doc's explicit sequencing (pretraining sanity check, forecasting-beats-baseline, causal-beats-saliency, feedback-loop-improves-accuracy). Do not skip a checkpoint to save time; each one exists because a downstream stage silently produces garbage if the upstream one is broken.

### What "Done" Looks Like

The project is complete and defensible without a real household pilot if:
- [ ] All four ablations run and show expected degradation
- [ ] Causal attribution beats the saliency baseline on synthetic top-k accuracy
- [ ] Forecasting head beats the static baseline on the proxy task
- [ ] Dashboard demonstrates the full loop end-to-end on synthetic + public data
- [ ] Dataset limitations and the synthetic-only causal validation are documented plainly, not glossed over
