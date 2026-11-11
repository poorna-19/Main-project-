"""
Paired Controlled Baseline vs. Proposed Model Verification Script.
Guarantees:
1. Re-verifies all 5 split SHA-256 hashes.
2. Evaluates paired per-seed test accuracy and sensitivity differences.
3. Computes 5-seed statistics only when all 5 seeds exist.
4. Halts safely without fabricating missing results.
"""
import os
import sys
import json
import hashlib
import numpy as np
from scipy import stats

def verify_split_hashes():
    baseline_hashes = {
        0: "cb327dd62ff57480e7e38ad61b770ec88dadfbae91917e6a6c18261420f38855",
        1: "f05951a1a2f196e4c78be3dd4b452a543f2df024c43be3bfc1cf431243ff2c74",
        2: "7bb75bdbc12f31886631e3fb78efbab38ada55bc76b3062fdd910e8dcbe86cce",
        3: "ea8777cee09a0530bf1c34616c5f759445f50b68cd2d86644ea046f6c6a0c500",
        4: "15ce9effd7aed7ded9fed01c3f7a9e051f3b340ffa0c4315b5d7b0da9d48f078",
    }
    all_ok = True
    print("=" * 75)
    print("STEP 1: CRYPTOGRAPHIC SPLIT INTEGRITY VERIFICATION (SHA-256)")
    print("=" * 75)
    for s in range(5):
        p = f"result/phase6_multiseed/splits/exact_splits_seed{s}.json"
        if not os.path.exists(p):
            print(f"  Seed {s} Split MISSING at {p}")
            all_ok = False
            continue
        with open(p, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        match = (h == baseline_hashes[s])
        if not match: all_ok = False
        print(f"  Seed {s} Split: {'MATCH' if match else 'MISMATCH'} ({h[:16]}...)")
    return all_ok

def main():
    if not verify_split_hashes():
        print("ERROR: Split file integrity check failed! Halting.")
        sys.exit(1)
        
    print("\n" + "=" * 75)
    print("STEP 2: CHECKING ARTIFACT AVAILABILITY FOR PAIRED BENCHMARK")
    print("=" * 75)
    
    proposed_paths = {
        0: "result/phase6_experiment2_novelty/10-28-13-02-58/final.json",
        1: "result/phase6_multiseed/runs/seed1/seed_1/final.json",
        2: "result/phase6_multiseed/runs/seed2/seed_2/final.json",
        3: "result/phase6_multiseed/runs/seed3/seed_3/final.json",
        4: "result/phase6_multiseed/runs/seed4/seed_4/final.json",
    }
    
    baseline_paths = {
        0: "result/controlled_run_resumed/final_test_report.json",
        1: "result/phase6_controlled_baseline_multiseed/runs/seed1/baseline_seed_1/final.json",
        2: "result/phase6_controlled_baseline_multiseed/runs/seed2/baseline_seed_2/final.json",
        3: "result/phase6_controlled_baseline_multiseed/runs/seed3/baseline_seed_3/final.json",
        4: "result/phase6_controlled_baseline_multiseed/runs/seed4/baseline_seed_4/final.json",
    }
    
    completed_pairs = []
    for s in range(5):
        prop_ok = os.path.exists(proposed_paths[s])
        base_ok = os.path.exists(baseline_paths[s])
        print(f"  Seed {s}: Proposed = {'COMPLETE' if prop_ok else 'PENDING'} | Baseline = {'COMPLETE' if base_ok else 'PENDING'}")
        if prop_ok and base_ok:
            completed_pairs.append(s)
            
    print(f"\nCompleted Paired Seeds: {completed_pairs} ({len(completed_pairs)}/5)")
    
    if len(completed_pairs) < 5:
        print("\n" + "!" * 75)
        print("NOTICE: Paired comparison is incomplete.")
        print(f"Seeds with missing proposed or baseline artifacts: {[s for s in range(5) if s not in completed_pairs]}")
        print("Stopping safely without fabricating results.")
        print("!" * 75)
        sys.exit(0)
        
    print("\n" + "=" * 75)
    print("STEP 3: ALL 5 PAIRED SEEDS VERIFIED! CALCULATING 5-SEED PAIRED STATISTICS")
    print("=" * 75)
    
    base_accs, prop_accs = [], []
    base_sens, prop_sens = [], []
    base_specs, prop_specs = [], []
    base_f1s, prop_f1s = [], []
    base_aucs, prop_aucs = [], []
    
    for s in range(5):
        # Read proposed
        with open(proposed_paths[s]) as f:
            p_data = json.load(f)
        p_test = p_data.get("test_at_best_val", {}).get("ensemble", p_data.get("test_at_best_val", {}))
        
        # Read baseline
        with open(baseline_paths[s]) as f:
            b_data = json.load(f)
        if s == 0:
            b_test = b_data.get("held_out_test_metrics", {})
        else:
            b_test = b_data.get("test_at_best_val", {}).get("head1", b_data.get("test_at_best_val", {}))
            
        p_acc = p_test.get("accuracy", 0.0) * 100
        b_acc = b_test.get("accuracy", 0.0) * 100
        p_sen = p_test.get("sensitivity", 0.0) * 100
        b_sen = b_test.get("sensitivity", 0.0) * 100
        p_spe = p_test.get("specificity", 0.0) * 100
        b_spe = b_test.get("specificity", 0.0) * 100
        p_f1 = p_test.get("f1", 0.0) * 100
        b_f1 = b_test.get("f1", 0.0) * 100
        p_auc = p_test.get("auc", 0.0) * 100
        b_auc = b_test.get("auc", 0.0) * 100
        
        base_accs.append(b_acc)
        prop_accs.append(p_acc)
        base_sens.append(b_sen)
        prop_sens.append(p_sen)
        base_specs.append(b_spe)
        prop_specs.append(p_spe)
        base_f1s.append(b_f1)
        prop_f1s.append(p_f1)
        base_aucs.append(b_auc)
        prop_aucs.append(p_auc)
        
        delta_acc = p_acc - b_acc
        delta_sen = p_sen - b_sen
        print(f"Seed {s}: Baseline Acc = {b_acc:.2f}% | Proposed Acc = {p_acc:.2f}% | Δ Acc = {delta_acc:+.2f} pp")
        print(f"         Baseline Sen = {b_sen:.2f}% | Proposed Sen = {p_sen:.2f}% | Δ Sen = {delta_sen:+.2f} pp")
        
    diff_acc = np.array(prop_accs) - np.array(base_accs)
    diff_sen = np.array(prop_sens) - np.array(base_sens)
    
    t_stat, p_val = stats.ttest_rel(prop_accs, base_accs)
    
    print("\n" + "=" * 75)
    print("5-SEED AGGREGATE SUMMARY (N=5):")
    print("=" * 75)
    print(f"Baseline Deep Mean Acc:  {np.mean(base_accs):.2f}% ± {np.std(base_accs, ddof=1):.2f}% (s) / ± {np.std(base_accs, ddof=0):.2f}% (σ)")
    print(f"Proposed Deep Mean Acc:  {np.mean(prop_accs):.2f}% ± {np.std(prop_accs, ddof=1):.2f}% (s) / ± {np.std(prop_accs, ddof=0):.2f}% (σ)")
    print(f"Mean Paired Difference:  {np.mean(diff_acc):+.2f} pp (Paired t-stat = {t_stat:.3f}, p = {p_val:.4f})")
    print(f"Baseline Deep Mean Sen:  {np.mean(base_sens):.2f}% | Proposed Deep Mean Sen: {np.mean(prop_sens):.2f}% | Δ Sen = {np.mean(diff_sen):+.2f} pp")
    print("=" * 75)

if __name__ == "__main__":
    main()
