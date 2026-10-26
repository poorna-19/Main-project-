"""
Evaluate best validation checkpoint on the held-out test set
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from dataset.abide import load_abide_data
from dataset.dataloader import init_stratified_dataloader
from models.CDCo_MambaNet.cdcomnet import CDCo_MambaNet
from training.metrics import calculate_metrics

def evaluate():
    checkpoint_path = "result/controlled_run/10-26-14-51-19/best_model.pt"
    config_path = "result/controlled_run/10-26-14-51-19/config.yaml"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
        
    cfg = OmegaConf.load(config_path)
    
    # Load dataset
    final_timeseires, final_pearson, labels, smri = load_abide_data(cfg)
    dataloaders = init_stratified_dataloader(cfg, final_timeseires, final_pearson, labels, smri)
    train_loader, val_loader, test_loader = dataloaders
    
    # Instantiate model
    model = CDCo_MambaNet(cfg).cuda()
    state_dict = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    
    # Evaluate on held-out test set ONLY ONCE
    pred_list = []
    label_list = []
    with torch.no_grad():
        for fMRI_Feature, fMRI_Con, label, sMRI_Feature in test_loader:
            fMRI_Feature = fMRI_Feature.cuda()
            fMRI_Con = fMRI_Con.cuda()
            label = label.cuda()
            sMRI_Feature = sMRI_Feature.cuda()
            
            output, _ = model(fMRI_Feature, fMRI_Con, sMRI_Feature)
            pred_list.extend(output.tolist())
            label_list.extend(label.tolist())
            
    metrics = calculate_metrics(pred_list, label_list, return_dict=True)
    print("HELD-OUT TEST SET EVALUATION RESULTS:")
    print(json.dumps(metrics, indent=2))
    
    # Also evaluate on validation set to verify
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for fMRI_Feature, fMRI_Con, label, sMRI_Feature in val_loader:
            fMRI_Feature = fMRI_Feature.cuda()
            fMRI_Con = fMRI_Con.cuda()
            label = label.cuda()
            sMRI_Feature = sMRI_Feature.cuda()
            
            output, _ = model(fMRI_Feature, fMRI_Con, sMRI_Feature)
            val_preds.extend(output.tolist())
            val_labels.extend(label.tolist())
            
    val_metrics = calculate_metrics(val_preds, val_labels, return_dict=True)
    print("VALIDATION SET VERIFICATION RESULTS:")
    print(json.dumps(val_metrics, indent=2))
    
    # Save test results to result/controlled_run/10-26-14-51-19/test_evaluation.json
    out_file = "result/controlled_run/10-26-14-51-19/test_evaluation.json"
    with open(out_file, "w") as f:
        json.dump({"test_metrics": metrics, "val_metrics": val_metrics}, f, indent=2)
    print(f"Saved evaluation results to {out_file}")

if __name__ == '__main__':
    evaluate()
