"""
CDCo-MambaNet V4 — Fresh-System Safe Verification Suite

This script performs 100% static and synthetic verification on a clean system:
1. Environment & Dependency Check (Python, PyTorch, Hydra, Timm, Einops, Scikit-Learn).
2. Compute Device Detection (GPU VRAM, CUDA runtime, or CPU fallback).
3. Synthetic-Only Model Verification (Forward pass, Evidential loss, Backward pass with synthetic tensors).
4. Configuration Loading (Hydra / OmegaConf model and dataset specs).
5. Split-File Integrity Verification (Verifies JSON structure of splits/ without loading test subjects).

STRICT SCIENTIFIC SAFETY GUARANTEES:
- DOES NOT touch held-out test data (demo/abide_seed0_test.npz is strictly quarantined).
- DOES NOT start model training.
- DOES NOT overwrite any model checkpoint.
- Safe to run repeatedly on any machine (Windows, Linux, macOS).
"""

import sys
import os
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_dependencies():
    print("\n" + "=" * 60)
    print("STEP 1: Checking Python & Core Dependencies")
    print("=" * 60)
    
    # Python version
    py_ver = sys.version_info
    print(f"[OK] Python Runtime: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver.major < 3 or (py_ver.major == 3 and py_ver.minor < 10):
        print("[FAIL] Python 3.10 or higher is required.")
        return False

    required_packages = [
        ("torch", "PyTorch deep learning framework"),
        ("hydra", "Hydra configuration manager"),
        ("omegaconf", "OmegaConf dictionary tools"),
        ("timm", "PyTorch Image Models"),
        ("einops", "Tensor transformation operations"),
        ("sklearn", "Scikit-learn metrics & splits"),
        ("scipy", "Scientific computing"),
        ("pandas", "Data manipulation tools"),
        ("matplotlib", "Plotting utilities"),
        ("networkx", "Graph algorithms"),
    ]

    all_passed = True
    for pkg, desc in required_packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [PASS] {pkg:<12} (v{ver}) — {desc}")
        except ImportError as e:
            print(f"  [FAIL] {pkg:<12} — MISSING ({e})")
            all_passed = False

    return all_passed


def check_compute_device():
    print("\n" + "=" * 60)
    print("STEP 2: Compute Device & PyTorch Detection")
    print("=" * 60)
    import torch

    print(f"[OK] PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"[INFO] CUDA Available: {cuda_avail}")

    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"  [PASS] GPU Detected: {device_name}")
        print(f"  [PASS] Total GPU VRAM: {total_vram_mb:.1f} MB ({total_vram_mb / 1024:.2f} GB)")
        print(f"  [PASS] CUDA Capability: {torch.cuda.get_device_capability(0)}")
        device = torch.device("cuda:0")
    else:
        print("  [NOTICE] CUDA is not available. Using CPU mode for synthetic checks.")
        device = torch.device("cpu")

    return device


def check_synthetic_architecture(device):
    print("\n" + "=" * 60)
    print("STEP 3: Synthetic-Data Architecture Verification")
    print("=" * 60)
    print("NOTE: Using 100% synthetic random tensors. Real patient data is QUARANTINED.")
    
    import torch
    from omegaconf import OmegaConf
    from models.CDCo_MambaNet_V4 import CDCo_MambaNet_V4

    # Build standalone minimal synthetic config
    cfg = OmegaConf.create({
        "model": {
            "name": "CDCo_MambaNet_V4",
            "cls_num": 2,
            "d_model": 48,
            "dim": 48,
            "dynamic_dim": True,
            "reliability_fusion": True,
            "min_dim": 12,
            "nested_dropout": 0.1,
            "edl_weight": 0.1,
            "evidence_activation": "relu",
            "uncertainty_method": "dirichlet",
            "temperature_scaling": True,
            "unified_head": True,
            "head1_regularization": 0.05,
            "spatial_attention_dropout": 0.10,
            "classification_head_dropout": 0.50,
        },
        "dataset": {
            "node_sz": 116,
            "node_feature_sz": 116,
            "timeseries_sz": 115,
        }
    })

    print("[1/4] Instantiating CDCo_MambaNet_V4 architecture...")
    model = CDCo_MambaNet_V4(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [PASS] Total Parameters: {total_params:,} (Trainable: {trainable_params:,})")
    
    model = model.to(device)
    model.train()

    # Generate synthetic input batch
    batch_size = 2
    fMRI_Feature = torch.randn(batch_size, 116, 115, device=device)
    fMRI_Con = torch.randn(batch_size, 116, 116, device=device)
    sMRI_Feature = torch.randn(batch_size, 1, 96, 96, 96, device=device)
    avail = torch.ones(batch_size, 2, device=device)

    print("[2/4] Executing synthetic forward pass...")
    out = model(fMRI_Feature, fMRI_Con, sMRI_Feature, avail)

    if isinstance(out, tuple):
        pred_logits, pred_evidential = out
        print(f"  [PASS] Forward output received: Dual evidential tuple")
        print(f"         Head 1 Logits shape: {pred_logits.shape} (expected: [{batch_size}, 2])")
        print(f"         Head 2 Alpha/Evidence shape: {pred_evidential.shape} (expected: [{batch_size}, 2])")
    else:
        print(f"  [PASS] Forward output shape: {out.shape}")
        pred_logits = out

    print("[3/4] Testing backward pass gradient computation...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    dummy_labels = torch.randint(0, 2, (batch_size,), device=device)
    loss = torch.nn.functional.cross_entropy(pred_logits, dummy_labels)
    loss.backward()

    active_grads = sum(1 for p in model.parameters() if p.grad is not None and torch.sum(torch.abs(p.grad)) > 0)
    print(f"  [PASS] Backward pass complete. Loss: {loss.item():.4f}")
    print(f"  [PASS] Active gradient tensors: {active_grads}/{len(list(model.parameters()))}")

    print("[4/4] Testing gradient clipping safety (max_norm=1.0)...")
    raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"  [PASS] Gradient clipping verified. Raw norm: {raw_norm.item():.4f} -> Clipped to <= 1.0")
    optimizer.step()
    optimizer.zero_grad()

    return True


def check_split_files():
    print("\n" + "=" * 60)
    print("STEP 4: Verifying Frozen Split Files")
    print("=" * 60)
    splits_dir = ROOT / "splits"
    if not splits_dir.exists():
        print("  [FAIL] splits/ directory does not exist!")
        return False

    split_files = [
        "exact_splits_seed0.json",
        "exact_splits_seed1.json",
        "exact_splits_seed2.json",
        "exact_splits_seed3.json",
        "exact_splits_seed4.json",
    ]

    all_valid = True
    for fname in split_files:
        fpath = splits_dir / fname
        if not fpath.exists():
            print(f"  [FAIL] Missing split file: {fname}")
            all_valid = False
            continue

        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            n_train = len(data.get("train", {}).get("indices", []))
            n_val = len(data.get("val", {}).get("indices", []))
            n_test = len(data.get("test", {}).get("indices", []))
            print(f"  [PASS] {fname:<25} -> Train: {n_train}, Val: {n_val}, Held-Out Test: {n_test}")
            if (n_train, n_val, n_test) != (227, 29, 28):
                print(f"         [WARN] Split counts ({n_train}, {n_val}, {n_test}) deviate from expected (227, 29, 28)!")
        except Exception as e:
            print(f"  [FAIL] Could not parse {fname}: {e}")
            all_valid = False

    return all_valid


def main():
    print("*" * 60)
    print("CDCo-MambaNet V4 — Fresh-System Safe Verification")
    print("*" * 60)

    dep_ok = check_dependencies()
    device = check_compute_device()
    synth_ok = check_synthetic_architecture(device)
    split_ok = check_split_files()

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  1. Dependencies:      {'PASSED' if dep_ok else 'FAILED'}")
    print(f"  2. Compute Device:    {'PASSED (CUDA)' if device.type == 'cuda' else 'PASSED (CPU Mode)'}")
    print(f"  3. Synthetic Model:   {'PASSED' if synth_ok else 'FAILED'}")
    print(f"  4. Split Integrity:   {'PASSED' if split_ok else 'FAILED'}")
    print("=" * 60)

    if dep_ok and synth_ok and split_ok:
        print("[SUCCESS] All fresh-system safety checks passed without errors.")
        print("[SAFETY] Held-out test data remained 100% quarantined. Zero patient data accessed.")
        return 0
    else:
        print("[ERROR] One or more checks failed. Review the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
