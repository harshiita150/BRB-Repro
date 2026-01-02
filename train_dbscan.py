import argparse
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
# These imports match your 'src' and 'config' folder structures
from src.datasets.dataset_init import get_train_eval_test_dataloaders
from src.deep import BRB_DEC, BRB_IDEC, BRB_DCN
from src.training.utils import set_cuda_configuration
from config.base_config import Args

# --- CLUSTERING ACCURACY (ACC) UTILITY ---
def cluster_acc(y_true, y_pred):
    """Calculates clustering accuracy using Hungarian matching."""
    y_true = y_true.astype(np.int64)
    mask = y_pred != -1 # Mask out DBSCAN noise
    if not np.any(mask): return 0.0
    y_p, y_t = y_pred[mask], y_true[mask]
    D = max(y_p.max(), y_t.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_p.size):
        w[y_p[i], y_t[i]] += 1
    ind = np.array(linear_sum_assignment(w.max() - w)).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_p.size

# --- DEEP DBSCAN MODEL WRAPPER ---
class DeepDBSCANWrapper(nn.Module):
    def __init__(self, base_model, feature_dim):
        super().__init__()
        self.encoder = base_model.encoder 
        self.classifier = nn.Linear(feature_dim, 10) # Initial head

    def forward(self, x):
        z = self.encoder(x)
        # Normalization is essential for DBSCAN epsilon stability
        z_norm = nn.functional.normalize(z, p=2, dim=1)
        logits = self.classifier(z_norm)
        return logits, z_norm

    def update_head(self, new_k, feature_dim):
        """Updates the linear head to match the number of clusters DBSCAN found."""
        device = next(self.parameters()).device
        self.classifier = nn.Linear(feature_dim, new_k).to(device)

def apply_soft_reset(model, alpha):
    """BRB Mechanism: Perturbs encoder weights to prevent stagnation."""
    for name, param in model.encoder.named_parameters():
        if 'weight' in name:
            noise = (torch.randn_like(param) * 0.02).to(param.device)
            param.data = (1 - alpha) * param.data + alpha * noise

# --- MAIN EXPERIMENT ENGINE ---
def run_master_experiment(custom_args):
    # FLEXIBLE DEVICE SELECTION
    if torch.cuda.is_available():
        set_cuda_configuration() # Call repo utility
        device = torch.device("cuda")
        print(f"🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("💻 GPU not detected. Running on CPU (Expect slower performance).")

    # LOAD DATASET (Supports all 8 repo datasets)
    repo_args = Args()
    repo_args.dataset = custom_args.dataset
    repo_args.batch_size = custom_args.batch_size
    train_loader, _, _ = get_train_eval_test_dataloaders(repo_args)
    
    # SELECT MODEL
    if custom_args.model_type == 'dec':
        base = BRB_DEC()
    elif custom_args.model_type == 'idec':
        base = BRB_IDEC()
    else:
        base = BRB_DCN()
        
    feature_dim = 128 # Standard latent dimension for your repo
    model = DeepDBSCANWrapper(base, feature_dim).to(device)
    
    results_log = []

    for p in range(custom_args.periods):
        print(f"\n--- {custom_args.dataset.upper()} | Period {p+1}/{custom_args.periods} ---")
        
        if p > 0:
            apply_soft_reset(model, custom_args.alpha)

        # 1. GENERATE LATENT FEATURES
        model.eval()
        all_z, all_y = [], []
        with torch.no_grad():
            for x, y in train_loader:
                _, z = model(x.to(device))
                all_z.append(z.cpu().numpy())
                all_y.append(y.numpy())
        
        X_feats = np.concatenate(all_z)
        Y_true = np.concatenate(all_y)

        # 2. DBSCAN RECLUSTERING
        print(f"Clustering with DBSCAN (eps={custom_args.eps})...")
        db = DBSCAN(eps=custom_args.eps, min_samples=custom_args.min_samples, n_jobs=-1).fit(X_feats)
        labels = torch.tensor(db.labels_).to(device)
        new_k = len(set(db.labels_) - {-1})
        print(f"Found {new_k} clusters | Noise: {np.mean(db.labels_ == -1):.1%}")

        if new_k < 2:
            print("Cluster collapse (found < 2 clusters). Try adjusting --eps.")
            continue

        model.update_head(new_k, feature_dim)
        optimizer = optim.Adam(model.parameters(), lr=custom_args.lr)
        criterion = nn.CrossEntropyLoss()

        # 3. FINE-TUNING ON PSEUDO-LABELS
        model.train()
        for epoch in range(custom_args.epochs):
            for i, (images, _) in enumerate(train_loader):
                batch_labels = labels[i*custom_args.batch_size : (i+1)*custom_args.batch_size]
                mask = batch_labels != -1 # Ignore DBSCAN noise
                
                if mask.any():
                    logits, _ = model(images.to(device))
                    loss = criterion(logits[mask], batch_labels[mask])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # 4. PERIOD EVALUATION
        acc = cluster_acc(Y_true, db.labels_)
        ari = metrics.adjusted_rand_score(Y_true, db.labels_)
        print(f">> Period ACC: {acc:.4f} | ARI: {ari:.4f}")
        results_log.append({'dataset': custom_args.dataset, 'period': p, 'acc': acc, 'ari': ari, 'k': new_k})

    # SAVE RESULTS
    output_name = f"results_{custom_args.dataset}_{custom_args.model_type}.csv"
    pd.DataFrame(results_log).to_csv(output_name, index=False)
    print(f"\n✅ Results saved to {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep DBSCAN + BRB Master Experiment")
    # All datasets from repo list
    parser.add_argument('--dataset', type=str, default='mnist', 
                        choices=['mnist', 'fmnist', 'kmnist', 'usps', 'gtsrb', 'optdigits', 'cifar10', 'cifar100-20'])
    parser.add_argument('--model_type', type=str, default='dec', choices=['dec', 'idec', 'dcn'])
    parser.add_argument('--eps', type=float, default=0.4, help="DBSCAN neighborhood distance.")
    parser.add_argument('--min_samples', type=int, default=10, help="Min points for a cluster.")
    parser.add_argument('--periods', type=int, default=5, help="Number of BRB soft-reset cycles.")
    parser.add_argument('--epochs', type=int, default=10, help="Training epochs per period.")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=0.1, help="Soft reset strength.")
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    run_master_experiment(args)
