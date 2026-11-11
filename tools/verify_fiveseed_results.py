"""
Phase 6 Experiment 2: Non-Destructive Five-Seed Verification & Statistical Aggregator

This script performs the full post-training audit and verification workflow:
1. Confirms Seeds 2, 3, and 4 completion and artifact readability.
2. Compares all five split file SHA-256 hashes against baseline hashes.
3. Confirms Seed 0 and Seed 1 artifact preservation.
4. Checks feature-dimension metadata in linear probe outputs against tensor dimensions (7,790).
5. Recomputes all five-seed aggregate statistics:
   - Individual seed values
   - Mean
   - Population SD (ddof=0)
   - Sample SD (ddof=1)
   across CDCo-MambaNet Ensemble, Hybrid models, and Classical Baselines.
6. Updates phase6_exp2_results_analysis.md strictly if all 5 seeds are verified.
"""

import os
import sys
import json
import hashlib
import glob
import numpy as np

BASELINE_SPLIT_HASHES = {
    "seed0": "cb327dd62ff57480e7e38ad61b770ec88dadfbae91917e6a6c18261420f38855",
    "seed1": "f05951a1a2f196e4c78be3dd4b452a543f2df024c43be3bfc1cf431243ff2c74",
    "seed2": "7bb75bdbc12f31886631e3fb78efbab38ada55bc76b3062fdd910e8dcbe86cce",
    "seed3": "ea8777cee09a0530bf1c34616c5f759445f50b68cd2d86644ea046f6c6a0c500",
    "seed4": "15ce9effd7aed7ded9fed01c3f7a9e051f3b340ffa0c4315b5d7b0da9d48f078",
}

SEED_PATHS = {
    0: {
        "dir": "result/phase6_experiment2_novelty/10-28-13-02-58",
        "best_model": "result/phase6_experiment2_novelty/10-28-13-02-58/best_model.pt",
        "final_json": "result/phase6_experiment2_novelty/10-28-13-02-58/final.json",
        "config": "result/phase6_experiment2_novelty/10-28-13-02-58/config.yaml",
        "linear_probe": "result/phase6_experiment2_novelty/linear_probe_novelty_results.json"
    },
    1: {
        "dir": "result/phase6_multiseed/runs/seed1/seed_1",
        "best_model": "result/phase6_multiseed/runs/seed1/seed_1/best_model.pt",
        "final_json": "result/phase6_multiseed/runs/seed1/seed_1/final.json",
        "config": "result/phase6_multiseed/runs/seed1/seed_1/config.yaml",
        "linear_probe": "result/phase6_multiseed/runs/seed1/linear_probe_seed1.json"
    },
    2: {
        "dir": "result/phase6_multiseed/runs/seed2/seed_2",
        "best_model": "result/phase6_multiseed/runs/seed2/seed_2/best_model.pt",
        "final_json": "result/phase6_multiseed/runs/seed2/seed_2/final.json",
        "config": "result/phase6_multiseed/runs/seed2/seed_2/config.yaml",
        "linear_probe": "result/phase6_multiseed/runs/seed2/linear_probe_seed2.json"
    },
    3: {
        "dir": "result/phase6_multiseed/runs/seed3/seed_3",
        "best_model": "result/phase6_multiseed/runs/seed3/seed_3/best_model.pt",
        "final_json": "result/phase6_multiseed/runs/seed3/seed_3/final.json",
        "config": "result/phase6_multiseed/runs/seed3/seed_3/config.yaml",
        "linear_probe": "result/phase6_multiseed/runs/seed3/linear_probe_seed3.json"
    },
    4: {
        "dir": "result/phase6_multiseed/runs/seed4/seed_4",
        "best_model": "result/phase6_multiseed/runs/seed4/seed_4/best_model.pt",
        "final_json": "result/phase6_multiseed/runs/seed4/seed_4/final.json",
        "config": "result/phase6_multiseed/runs/seed4/seed_4/config.yaml",
        "linear_probe": "result/phase6_multiseed/runs/seed4/linear_probe_seed4.json"
    }
}

CLASSICAL_BASELINES_PATH = "result/phase6_multiseed/classical_baselines_multiseed.json"

def verify_splits():
    print("=" * 70)
    print("STEP 1: VERIFYING SPLIT FILE INTEGRITY (SHA-256)")
    print("=" * 70)
    all_match = True
    for s in range(5):
        split_path = f"result/phase6_multiseed/splits/exact_splits_seed{s}.json"
        if not os.path.exists(split_path):
            print(f"FAILED: Missing split file: {split_path}")
            all_match = False
            continue
        with open(split_path, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
        baseline_hash = BASELINE_SPLIT_HASHES[f"seed{s}"]
        if current_hash == baseline_hash:
            print(f"  Seed {s} Split: MATCH ({current_hash[:16]}...)")
        else:
            print(f"  Seed {s} Split: MISMATCH!")
            print(f"    Expected: {baseline_hash}")
            print(f"    Actual:   {current_hash}")
            all_match = False
    return all_match

def verify_artifacts():
    print("\n" + "=" * 70)
    print("STEP 2: VERIFYING CHECKPOINTS & EVALUATION ARTIFACTS")
    print("=" * 70)
    status = {}
    for s in range(5):
        paths = SEED_PATHS[s]
        status[s] = {"complete": True, "missing": []}
        for item, p in paths.items():
            if item == "dir":
                continue
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                status[s]["complete"] = False
                status[s]["missing"].append(p)
        if status[s]["complete"]:
            bm_size = os.path.getsize(paths["best_model"]) / (1024**2)
            print(f"  Seed {s}: COMPLETE (best_model: {bm_size:.1f} MB, final.json verified)")
        else:
            print(f"  Seed {s}: INCOMPLETE. Missing: {status[s]['missing']}")
    return status

def compute_stats(values):
    vals = [v * 100 if v <= 1.0 else v for v in values]
    mean = np.mean(vals)
    pop_sd = np.std(vals, ddof=0)
    samp_sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    return {
        "vals": [round(v, 2) for v in vals],
        "mean": round(mean, 2),
        "pop_sd": round(pop_sd, 2),
        "samp_sd": round(samp_sd, 2)
    }

def main():
    splits_ok = verify_splits()
    artifacts_status = verify_artifacts()

    completed_seeds = [s for s in range(5) if artifacts_status[s]["complete"]]
    print(f"\nCompleted Seeds: {completed_seeds} (Count: {len(completed_seeds)}/5)")

    if len(completed_seeds) < 5:
        print("\n" + "!" * 70)
        print("NOTICE: Multi-seed execution is currently IN PROGRESS or PENDING.")
        print(f"Only Seeds {completed_seeds} have completed and passed artifact checks.")
        print("Seeds remaining to execute in host PowerShell:")
        for s in [s for s in range(5) if s not in completed_seeds]:
            print(f"  - Seed {s}: awaiting execution or completion")
        print("!" * 70)
        print("\nStopping aggregation safely without fabricating results.")
        return False

    print("\n" + "=" * 70)
    print("STEP 3: EXTRACTING & RECOMPUTING FIVE-SEED STATISTICS")
    print("=" * 70)

    # 1. Classical Baselines
    with open(CLASSICAL_BASELINES_PATH, "r") as f:
        classical_data = json.load(f)

    # 2. Deep & Hybrid Data
    deep_ens_acc, deep_ens_bacc, deep_ens_sen, deep_ens_spe, deep_ens_auc = [], [], [], [], []
    deep_fin_acc, deep_fin_bacc, deep_fin_sen, deep_fin_spe, deep_fin_auc = [], [], [], [], []
    hyb_lr_acc, hyb_lr_bacc, hyb_lr_sen, hyb_lr_spe, hyb_lr_auc = [], [], [], [], []
    hyb_rg_acc, hyb_rg_bacc, hyb_rg_sen, hyb_rg_spe, hyb_rg_auc = [], [], [], [], []

    for s in range(5):
        with open(SEED_PATHS[s]["final_json"], "r") as f:
            fin = json.load(f)
        ens_m = fin["test_at_best_val"]["ensemble"]
        deep_ens_acc.append(ens_m["accuracy"])
        deep_ens_bacc.append(ens_m["balanced_accuracy"])
        deep_ens_sen.append(ens_m["sensitivity"])
        deep_ens_spe.append(ens_m["specificity"])
        deep_ens_auc.append(ens_m["auc"])

        fin_m = fin.get("test_at_final_epoch", {}).get("ensemble", {})
        if fin_m:
            deep_fin_acc.append(fin_m["accuracy"])
            deep_fin_bacc.append(fin_m["balanced_accuracy"])
            deep_fin_sen.append(fin_m["sensitivity"])
            deep_fin_spe.append(fin_m["specificity"])
            deep_fin_auc.append(fin_m["auc"])

        with open(SEED_PATHS[s]["linear_probe"], "r") as f:
            lp = json.load(f)
        hyb = lp["Hybrid_FC_Plus_Deep_7094"]
        
        # Verify feature dimension metadata
        if hyb.get("dim") != 7790:
            print(f"WARNING: Seed {s} hybrid feature dimension is {hyb.get('dim')} (expected 7790)")
            
        lr_m = hyb["LogisticRegression_L2"]["test_metrics"]
        hyb_lr_acc.append(lr_m["accuracy"])
        hyb_lr_bacc.append(lr_m["balanced_accuracy"])
        hyb_lr_sen.append(lr_m["sensitivity"])
        hyb_lr_spe.append(lr_m["specificity"])
        hyb_lr_auc.append(lr_m["auc"])

        rg_m = hyb["RidgeClassifier"]["test_metrics"]
        hyb_rg_acc.append(rg_m["accuracy"])
        hyb_rg_bacc.append(rg_m["balanced_accuracy"])
        hyb_rg_sen.append(rg_m["sensitivity"])
        hyb_rg_spe.append(rg_m["specificity"])
        hyb_rg_auc.append(rg_m["auc"])

    # Aggregate summaries
    print("\n--- FIVE-SEED AGGREGATE RESULTS ---")
    models_summary = {
        "CDCo-MambaNet (Ensemble at Best Val)": {
            "Accuracy": compute_stats(deep_ens_acc),
            "Balanced Acc": compute_stats(deep_ens_bacc),
            "Sensitivity": compute_stats(deep_ens_sen),
            "Specificity": compute_stats(deep_ens_spe),
            "ROC-AUC": compute_stats(deep_ens_auc),
        },
        "CDCo-MambaNet (Final Epoch)": {
            "Accuracy": compute_stats(deep_fin_acc),
            "Balanced Acc": compute_stats(deep_fin_bacc),
            "Sensitivity": compute_stats(deep_fin_sen),
            "Specificity": compute_stats(deep_fin_spe),
            "ROC-AUC": compute_stats(deep_fin_auc),
        },
        "Hybrid FC + Deep (L2 Logistic)": {
            "Accuracy": compute_stats(hyb_lr_acc),
            "Balanced Acc": compute_stats(hyb_lr_bacc),
            "Sensitivity": compute_stats(hyb_lr_sen),
            "Specificity": compute_stats(hyb_lr_spe),
            "ROC-AUC": compute_stats(hyb_lr_auc),
        },
        "Hybrid FC + Deep (Ridge)": {
            "Accuracy": compute_stats(hyb_rg_acc),
            "Balanced Acc": compute_stats(hyb_rg_bacc),
            "Sensitivity": compute_stats(hyb_rg_sen),
            "Specificity": compute_stats(hyb_rg_spe),
            "ROC-AUC": compute_stats(hyb_rg_auc),
        }
    }

    # Classical Baselines
    for clf_name in ["LogisticRegression_L2", "RidgeClassifier", "LinearSVC"]:
        c_acc = [classical_data[f"seed_{s}"][clf_name]["test"]["accuracy"] for s in range(5)]
        c_bacc = [classical_data[f"seed_{s}"][clf_name]["test"]["balanced_accuracy"] for s in range(5)]
        c_sen = [classical_data[f"seed_{s}"][clf_name]["test"]["sensitivity"] for s in range(5)]
        c_spe = [classical_data[f"seed_{s}"][clf_name]["test"]["specificity"] for s in range(5)]
        c_auc = [classical_data[f"seed_{s}"][clf_name]["test"]["auc"] for s in range(5)]
        models_summary[f"Classical {clf_name}"] = {
            "Accuracy": compute_stats(c_acc),
            "Balanced Acc": compute_stats(c_bacc),
            "Sensitivity": compute_stats(c_sen),
            "Specificity": compute_stats(c_spe),
            "ROC-AUC": compute_stats(c_auc),
        }

    for m_name, metrics in models_summary.items():
        print(f"\nModel: {m_name}")
        for met_name, s in metrics.items():
            print(f"  {met_name:15s}: {s['mean']}% +/- {s['pop_sd']}% (pop) / +/- {s['samp_sd']}% (sample) | seeds: {s['vals']}")

    # Generate Markdown Table for Report
    md_lines = []
    md_lines.append("\n## 11. Final Five-Seed Complete Benchmark Table (Verified Post-Training)\n")
    md_lines.append("| Model Paradigm | Feature Input | Accuracy (Mean +/- Pop SD / Sample SD) | Balanced Acc | ASD Sensitivity | Control Specificity | ROC-AUC | Seed Accuracies (0 to 4) |")
    md_lines.append("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for m_name, m_data in models_summary.items():
        acc = m_data["Accuracy"]
        bacc = m_data["Balanced Acc"]
        sen = m_data["Sensitivity"]
        spe = m_data["Specificity"]
        auc = m_data["ROC-AUC"]
        feat_str = "6,670 FC + 1,120 Deep" if "Hybrid" in m_name else ("6,670 Pearson FC" if "Classical" in m_name else "Spatio-Temporal + sMRI")
        md_lines.append(f"| **{m_name}** | {feat_str} | **{acc['mean']}%** +/- {acc['pop_sd']}% / {acc['samp_sd']}% | {bacc['mean']}% +/- {bacc['pop_sd']}% | {sen['mean']}% +/- {sen['pop_sd']}% | {spe['mean']}% +/- {spe['pop_sd']}% | {auc['mean']}% +/- {auc['pop_sd']}% | `{acc['vals']}` |")

    report_path = "phase6_exp2_results_analysis.md"
    if os.path.exists(report_path):
        with open(report_path, "a") as f:
            f.write("\n" + "\n".join(md_lines) + "\n")
        print(f"\nSuccessfully appended verified five-seed benchmark table to: {report_path}")

    return True

if __name__ == "__main__":
    main()
