import argparse
from argparse import Namespace # Added to fix the Args() TypeError
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import os

# --- REPO-SPECIFIC IMPORTS ---
from src.datasets.dataset_init import get_train_eval_test_dataloaders
from src.deep import BRB_DEC, BRB_IDEC, BRB_DCN
from src.training.utils import set_cuda_configuration
# We keep this import in case other parts of the repo need the class definition
from config.base_config import Args 

# --- CLUSTERING ACCURACY (ACC) ---
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    mask = y_pred != -1 
    if not np.any(mask): return 0.0
    y_p, y_t = y_pred[mask], y_true[mask]
    D = max(y_p.max(), y_t.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_p.size):
        w[y_p[i], y_t[i]] += 1
    ind = np.array(linear_sum_assignment(w.max() - w)).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_p.size

# --- MODEL WRAPPER ---
class DeepDBSCANWrapper(nn.Module):
    def __init__(self, base_model, feature_dim):
        super().__init__()
        self.encoder = base_model.encoder 
        self.classifier = nn.Linear(feature_dim, 10) 

    def forward(self, x):
        z = self.encoder(x)
        z_norm = nn.functional.normalize(z, p=2, dim=1)
        logits = self.classifier(z_norm)
        return logits, z_norm

    def update_head(self, new_k, feature_dim):
        device = next(self.parameters()).device
        self.classifier = nn.Linear(feature_dim, new_k).to(device)

def apply_soft_reset(model, interpolation_factor):
    """BRB Mechanism: Perturbs weights."""
    # Check if model is wrapped in DataParallel for multi-GPU
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    for name, param in target_model.encoder.named_parameters():
        if 'weight' in name:
            noise = (torch.randn_like(param) * 0.02).to(param.device)
            param.data = interpolation_factor * param.data + (1 - interpolation_factor) * noise

# --- EXPERIMENT ENGINE ---
def run_experiment(args):
    # FIX 1: Handle "all" GPU safely to avoid ValueError
    if torch.cuda.is_available():
        if args.experiment_gpu != 'all':
            # Only call the repo utility for specific IDs (0, 1, etc.)
            set_cuda_configuration(args.experiment_gpu) 
        device = torch.device("cuda")
        print(f"🚀 Using GPU(s). Count: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")

    # FIX 2: Use Namespace to avoid 'Args() missing 4 arguments' TypeError
    # The dataloader typically only needs 'dataset' and 'batch_size' from this object.
    repo_args = Namespace(
        dataset=args.dataset_name,
        batch_size=args.batch_size
    )
    train_loader, _, _ = get_train_eval_test_dataloaders(repo_args)

    # Auto-calculate periods
    num_periods = args.clustering_epochs // args.brb_reset_interval
    
    # Model Selection
    if args.dc_algorithm == 'dec':
        base = BRB_DEC()
    elif args.dc_algorithm == 'idec':
        base = BRB_IDEC()
    else:
        base = BRB_DCN()
        
    feature_dim = 128 
    model = DeepDBSCANWrapper(base, feature_dim).to(device)
    
    # Multi-GPU support if "all" is requested
    if torch.cuda.device_count() > 1 and args.experiment_gpu == 'all':
        print(f"🔗 Multi-GPU Active: Training on {torch.cuda.device_count()} cards.")
        model = nn.DataParallel(model)

    results_log = []

    for p in range(num_periods):
        print(f"\n--- Period {p+1}/{num_periods} ---")
        if p > 0:
            apply_soft_reset(model, args.brb_reset_interpolation_factor)

        model.eval()
        all_z, all_y = [], []
        with torch.no_grad():
            for x, y in train_loader:
                _, z = model(x.to(device))
                all_z.append(z.cpu().numpy())
                all_y.append(y.numpy())
        
        X_feats = np.concatenate(all_z)
        Y_true = np.concatenate(all_y)

        db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1).fit(X_feats)
        labels = torch.tensor(db.labels_).to(device)
        new_k = len(set(db.labels_) - {-1})

        if new_k < 2:
            print("Cluster collapse. Increase --dbscan-eps.")
            continue

        # Update classifier head based on found clusters
        if isinstance(model, nn.DataParallel):
            model.module.update_head(new_k, feature_dim)
        else:
            model.update_head(new_k, feature_dim)
            
        optimizer = optim.Adam(model.parameters(), lr=args.dc_optimizer_lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(args.brb_reset_interval):
            for i, (images, _) in enumerate(train_loader):
                batch_labels = labels[i*args.batch_size : (i+1)*args.batch_size]
                mask = batch_labels != -1
                if mask.any():
                    logits, _ = model(images.to(device))
                    loss = criterion(logits[mask], batch_labels[mask])
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

        acc = cluster_acc(Y_true, db.labels_)
        ari = metrics.adjusted_rand_score(Y_true, db.labels_)
        print(f">> ACC: {acc:.4f} | ARI: {ari:.4f} | K: {new_k}")
        results_log.append({'period': p, 'acc': acc, 'ari': ari})

    pd.DataFrame(results_log).to_csv(f"results_{args.dataset_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Parameters matching your requested style
    parser.add_argument('--dataset-name', type=str, default='mnist', dest='dataset_name')
    parser.add_argument('--experiment.gpu', type=str, default='all', dest='experiment_gpu')
    parser.add_argument('--dc-algorithm', type=str, default='dec', dest='dc_algorithm')
    parser.add_argument('--batch-size', type=int, default=256, dest='batch_size')
    parser.add_argument('--dc-optimizer.lr', type=float, default=0.001, dest='dc_optimizer_lr')
    
    # BRB logic parameters
    parser.add_argument('--clustering-epochs', type=int, default=100, dest='clustering_epochs')
    parser.add_argument('--brb.reset-interval', type=int, default=20, dest='brb_reset_interval')
    parser.add_argument('--brb.reset-interpolation-factor', type=float, default=0.8, dest='brb_reset_interpolation_factor')
    
    # DBSCAN clustering parameters
    parser.add_argument('--dbscan-eps', type=float, default=0.4, dest='dbscan_eps')
    parser.add_argument('--dbscan-min-samples', type=int, default=10, dest='dbscan_min_samples')
    
    args = parser.parse_args()
    run_experiment(args)
