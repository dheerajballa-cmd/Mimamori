import os
import json
import random
from typing import Dict, List, Tuple

def generate_splits(subject_ids: List[str], train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> Dict[str, List[str]]:
    """Generates train/val/test splits deterministically based on subject IDs."""
    random.seed(seed)
    shuffled = sorted(list(subject_ids)) # Sort first to ensure deterministic shuffle across systems
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:]
    }

def get_frozen_splits() -> Dict[str, Dict[str, List[str]]]:
    """Returns frozen subject-level train/val/test splits for all datasets."""
    # We construct them dynamically based on available data, using a fixed seed.
    import sys
    sys.path.append("D:/Mimamori")
    from data.loaders.loaders import (
        SisFallLoader,
        KFallLoader,
        EHomeSeniorsLoader,
        SoundsOfHomeLoader,
        SyntheticLongitudinalLoader
    )
    
    splits = {}
    seed = 42
    
    # 1. SisFall
    try:
        sf_loader = SisFallLoader()
        sf_subs = sf_loader.list_subjects()
        splits["SisFall"] = generate_splits(sf_subs, seed=seed)
    except Exception as e:
        print(f"Failed to split SisFall: {e}")
        
    # 2. KFall
    try:
        kf_loader = KFallLoader()
        kf_subs = kf_loader.list_subjects(elderly=False)
        splits["KFall"] = generate_splits(kf_subs, seed=seed)
    except Exception as e:
        print(f"Failed to split KFall: {e}")
        
    # 3. KFall Elderly
    try:
        kf_loader = KFallLoader()
        kfe_subs = kf_loader.list_subjects(elderly=True)
        splits["KFall_Elderly"] = generate_splits(kfe_subs, seed=seed)
    except Exception as e:
        print(f"Failed to split KFall_Elderly: {e}")
        
    # 4. Sounds of Home
    try:
        sh_loader = SoundsOfHomeLoader()
        sh_subs = [os.path.basename(s) for s in sh_loader.list_subject_dirs()]
        splits["Sounds_of_Home"] = generate_splits(sh_subs, seed=seed)
    except Exception as e:
        print(f"Failed to split Sounds_of_Home: {e}")
        
    # 5. Synthetic
    try:
        syn_loader = SyntheticLongitudinalLoader()
        parts = syn_loader.load_participants()
        syn_subs = parts["participant_id"].astype(str).tolist()
        splits["Synthetic"] = generate_splits(syn_subs, seed=seed)
    except Exception as e:
        print(f"Failed to split Synthetic: {e}")
        
    # eHomeSeniors and NHATS don't strictly use standard subject directories in loaders yet,
    # so we might skip their explicit ID split unless parsed.

    return splits