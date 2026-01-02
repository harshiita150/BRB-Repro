import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from argparse import Namespace

# --- REPO-SPECIFIC IMPORTS (Matches train.py) ---
from src.datasets.dataset_init import get_train_eval_test_dataloaders, _get_smallest_subset
from src.deep import BRB_DEC, BRB_IDEC, BRB_DCN
from src.training.ae_init import initialize_autoencoder
from src.training.utils import set_cuda_configuration, get_gpu_with_most_free_memory
from src.deep._torch_utils import set_torch_seed

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
    """BRB weight perturbation logic."""
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    for name, param in target_model.encoder.named_parameters():
        if 'weight' in name:
            noise = (torch.randn_like(param) * 0.02).to(param.device)
            param.data = interpolation_factor * param.data + (1 - interpolation_factor) * noise

# --- EXPERIMENT ENGINE ---
def run_experiment(user_args):
    # 1. SETUP PATHS
    result_dir = Path(".").resolve()
    data_path = result_dir / "data"

    # 2. SEEDING (Mimics train.py)
    rng = np.random.RandomState(42)
    set_torch_seed(rng)

    # 3. CONSTRUCT NESTED CONFIG (Crucial to fix AttributeErrors)
    # This structure satisfies the internal repo requirements
    args = Namespace(
        dataset_name=user_args.dataset_name,
        dataset_subset=_get_smallest_subset(user_args.dataset_name, "smallest"),
        batch_size=user_args.batch_size,
        convnet="none", # Fixes image_b572c2.png
        use_contrastive_loss=False,
        augmentation_invariance=False,
        n_clusters=None,
        embedding_dim=128,
        model_type="mlp",
        experiment=Namespace(
            gpu=user_args.gpu,
            num_workers_dataloader=4
        )
    )

    # 4. GPU LOGIC (Fixes image_b49ce3.png)
    if torch.cuda.is_available():
        if args.experiment.gpu == "all":
            device = torch.device("cuda")
            print(f"🚀 Using all available GPUs.")
        else:
            # Use the repo's logic to find the best specific GPU
            gpu = get_gpu_with_most_free_memory(args.experiment.gpu)
            device = torch.device(gpu)
    else:
        device = torch.device("cpu")

    # 5. DATALOADING (Fixes image_b501c8.png)
    # Passing both args and data_path as required by the function signature
    ae_train_dl, _, dc_train_dl, _, test_dl, data, labels, _, _ = get_train_eval_test_dataloaders(args, data_path)

    # 6. INITIALIZE MODEL (Using repo's autoencoder init)
    ae = initialize_autoencoder(args=args, data=data, device=device)
    feature_dim = 128
    model = DeepDBSCANWrapper(ae, feature_dim).to(device)

    if torch.cuda.device_count() > 1 and user_args.gpu == 'all':
        model = nn.DataParallel(model)

    results_log = []
    num_periods = user_args.epochs // user_args.reset_interval

    # 7. MAIN TRAINING LOOP
    for p in range(num_periods):
        print(f"\n--- Period {p+1}/{num_periods} ---")
        if p > 0:
            apply_soft_reset(model, user_args.alpha)

        # A. EXTRACT FEATURES
        model.eval()
        all_z, all_y = [], []
        with torch.no_grad():
            for x, y in dc_train_dl:
                _, z = model(x.to(device))
                all_z.append(z.cpu().numpy())
                all_y.append(y.numpy())
        
        X_feats = np.concatenate(all_z)
        Y_true = np.concatenate(all_y)

        # B. RUN DBSCAN
        db = DBSCAN(eps=user_args.eps, min_samples=10, n_jobs=-1).fit(X_feats)
        labels_found = torch.tensor(db.labels_).to(device)
        new_k = len(set(db.labels_) - {-1})

        if new_k < 2:
            print("Warning: DBSCAN found too few clusters. Try adjusting --eps.")
            continue

        # C. UPDATE HEAD & TRAIN
        if isinstance(model, nn.DataParallel):
            model.module.update_head(new_k, feature_dim)
        else:
            model.update_head(new_k, feature_dim)
            
        optimizer = optim.Adam(model.parameters(), lr=user_args.lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(user_args.reset_interval):
            for i, (images, _) in enumerate(dc_train_dl):
                batch_labels = labels_found[i*user_args.batch_size : (i+1)*user_args.batch_size]
                mask = batch_labels != -1
                if mask.any():
                    logits, _ = model(images.to(device))
                    loss = criterion(logits[mask], batch_labels[mask])
                    optimizer.zero_grad(); loss.backward(); optimizer.step()

        acc = cluster_acc(Y_true, db.labels_)
        print(f">> Period {p+1} Result | ACC: {acc:.4f} | Found K: {new_k}")
        results_log.append({'period': p, 'acc': acc, 'k': new_k})

    pd.DataFrame(results_log).to_csv(f"dbscan_results_{user_args.dataset_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='mnist')
    parser.add_argument('--gpu', type=str, default='all')
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--reset-interval', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=0.8) # Reset factor
    
    args = parser.parse_args()
    run_experiment(args)
