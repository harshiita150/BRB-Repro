import warnings
warnings.filterwarnings("ignore", message=".*Deterministic behavior.*")
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from shortuuid import uuid
from threadpoolctl import threadpool_limits
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

import wandb
from config.base_config import Args
from src.datasets.dataset_init import _get_smallest_subset, get_train_eval_test_dataloaders
from src.deep._torch_utils import set_torch_seed
from src.deep.brb_reclustering import brb_settings_printout, brb_short_printout, soft_reset
from src.deep.evaluation import evaluate_deep_clustering
from src.deep._clustering_utils import encode_batchwise
from src.training._utils import determine_optimizer
from src.training.ae import pretrain_ae
from src.training.ae_init import initialize_autoencoder
from src.training.utils import get_ae_path, get_gpu_with_most_free_memory, get_number_of_clusters, set_cuda_configuration
from src.training.wandb_init import initialize_wandb

# --- CLUSTERING ACCURACY (ACC) ---
def cluster_acc(y_true, y_pred):
    """Calculate clustering accuracy using Hungarian algorithm."""
    y_true = y_true.astype(np.int64)
    mask = y_pred != -1 
    if not np.any(mask): 
        return 0.0
    y_p, y_t = y_pred[mask], y_true[mask]
    D = max(y_p.max(), y_t.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_p.size):
        w[y_p[i], y_t[i]] += 1
    ind = np.array(linear_sum_assignment(w.max() - w)).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_p.size

def cluster_acc_all(y_true, y_pred):
    """Calculates accuracy where noise (-1) is treated as an incorrect prediction."""
    y_true = y_true.astype(np.int64)
    y_pred_mapped = y_pred.copy()
    
    noise_id = max(y_true.max(), y_pred.max()) + 1
    y_pred_mapped[y_pred == -1] = noise_id
    
    D = max(y_pred_mapped.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_true.size):
        w[y_pred_mapped[i], y_true[i]] += 1
    
    ind = np.array(linear_sum_assignment(w.max() - w)).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_true.size


# --- MODEL WRAPPER FOR DBSCAN ---
class DeepDBSCANWrapper(nn.Module):
    """Wrapper that adds a classifier head to the autoencoder for DBSCAN-based clustering."""
    def __init__(self, base_model, feature_dim, n_clusters=10):
        super().__init__()
        self.encoder = base_model.encoder 
        self.classifier = nn.Linear(feature_dim, n_clusters) 

    def forward(self, x):
        z = self.encoder(x)
        z_norm = nn.functional.normalize(z, p=2, dim=1)
        logits = self.classifier(z_norm)
        return logits, z_norm

    def update_head(self, new_k, feature_dim):
        """Update the classifier head to match the number of clusters found by DBSCAN."""
        device = next(self.parameters()).device
        self.classifier = nn.Linear(feature_dim, new_k).to(device)


@threadpool_limits.wrap(limits=45, user_api="openmp")
@threadpool_limits.wrap(limits=1, user_api="blas")
def train_dbscan(args: Args) -> None:
    """Training script for DBSCAN with BRB integration.
    
    Similar to train.py but uses DBSCAN for clustering instead of KMeans-based methods.
    Integrates DBSCAN with BRB (Belief Rule Base) for weight reset and reclustering.
    """
    
    if args.experiment.result_dir is None:
        result_dir = Path(".").resolve()
    else:
        result_dir = Path(args.experiment.result_dir).resolve()

    # Infer the smallest subset for the dataset if "smallest" is specified
    args.dataset_subset = _get_smallest_subset(args.dataset_name, args.dataset_subset)
    print(f"Dataset subset: {args.dataset_subset}. Dataset name: {args.dataset_name}.")

    # Set some paths
    run_name = (
        f"{args.dataset_name}-dbscan-{args.model_type}-{args.dc_optimizer.lr}-"
        f"{args.brb.reset_weights}-{args.brb.recluster}-"
        f"{args.brb.reset_momentum}-{args.brb.recalculate_centers}-{uuid()}"
    )
    data_path = result_dir / "data"
    base_path = result_dir / f"experiments/{args.experiment.prefix}/{run_name}"
    wandb_logging_dir = result_dir

    cp_path = base_path / f"seed_{args.seed}"
    cp_path.mkdir(parents=True, exist_ok=True)
    if args.ae_path is None:
        args.ae_path = get_ae_path(args, data_path)
    print("Start: ", base_path)

    # Seeding & reproducibility
    rng = np.random.RandomState(args.seed)
    set_torch_seed(rng)
    torch.use_deterministic_algorithms(args.experiment.deterministic_torch, warn_only=True)
    torch.set_float32_matmul_precision("medium")

    # Set number of clusters to ground truth number of clusters if n_clusters is not specified
    if args.n_clusters is None:
        args.n_clusters = get_number_of_clusters(args.dataset_name, args.dataset_subset)

    # Set embedding dimension to number of clusters if not specified
    if args.embedding_dim is None:
        args.embedding_dim = args.n_clusters

    # Initialize wandb run
    run = initialize_wandb(args=args, name=run_name, wandb_logging_dir=wandb_logging_dir)

    # Load and preprocess data
    ae_train_dl, _, dc_train_dl, dc_train_wA_dl, test_dl, data, labels, _, test_labels = get_train_eval_test_dataloaders(
        args, data_path
    )

    # Sanity check
    assert args.n_clusters == len(set(labels.tolist()))

    # Device setup
    if isinstance(args.experiment.gpu, tuple | list | str):
        time.sleep(random.randint(0, 5))
        gpu = get_gpu_with_most_free_memory(args.experiment.gpu)
        device = torch.device(gpu)
        if args.experiment.track_wandb:
            run.log({"System/GPU": gpu})
    else:
        device = set_cuda_configuration(args.experiment.gpu)

    ae = initialize_autoencoder(args=args, data=data, device=device)

    # Print out general BRB settings before training
    brb_short_printout(vars(args.brb))
    brb_settings_printout(vars(args.brb))

    # Pretrain autoencoder
    ae = pretrain_ae(
        ae=ae,
        dataloader=ae_train_dl,
        optimizer_args=args.pretrain_optimizer,
        n_epochs=args.pretrain_epochs,
        overwrite_ae=args.overwrite_ae,
        ae_path=args.ae_path,
        save_model=args.save_ae,
        wandb_run=run,
        device=device,
    )

    # Set up the optimizer
    optimizer_class = determine_optimizer(
        optimizer_name=args.dc_optimizer.optimizer, weight_decay=args.dc_optimizer.weight_decay
    )

    dbscan_labels = None
    cluster_centers = None
    
    if args.clustering_epochs > 0:
        # Initialize DBSCAN wrapper model
        feature_dim = args.embedding_dim if args.embedding_dim is not None else 128
        model = DeepDBSCANWrapper(ae, feature_dim, n_clusters=args.n_clusters).to(device)
        
        # Setup optimizer
        optimizer = optimizer_class(model.parameters(), lr=args.dc_optimizer.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Calculate number of periods based on reset interval
        num_periods = args.clustering_epochs // args.brb.reset_interval
        if num_periods == 0:
            num_periods = 1
        
        metrics_log = []
        
        # Main training loop with BRB integration
        for period in range(num_periods):
            print(f"\n--- Period {period+1}/{num_periods} ---")
            
            # Apply BRB reset if not first period
            if period > 0:
                print(f"Applying BRB reset at period {period+1}")
                # Apply soft reset to autoencoder
                if args.brb.reset_weights:
                    soft_reset(
                        autoencoder=ae,
                        reset_interpolation_factor=args.brb.reset_interpolation_factor,
                        reset_interpolation_factor_step=args.brb.reset_interpolation_factor_step,
                        reset_batchnorm=args.brb.reset_batchnorm,
                        reset_embedding=args.brb.reset_embedding,
                        reset_projector=args.brb.reset_projector,
                        reset_convlayers=args.brb.reset_convlayers,
                    )
            
            # Extract features
            model.eval()
            all_z, all_indices = [], []
            with torch.no_grad():
                for batch in dc_train_wA_dl:
                    # Dataloader returns (index, data)
                    # batch[0] = indices, batch[1] = data
                    batch_indices = batch[0].numpy()
                    batch_data = batch[1].to(device)
                    _, z = model(batch_data)
                    all_z.append(z.cpu().numpy())
                    all_indices.append(batch_indices)
            
            X_feats = np.concatenate(all_z)
            all_indices = np.concatenate(all_indices)
            # Get ground truth labels using indices
            Y_true = labels[all_indices] if labels is not None else None
            
            # Run DBSCAN clustering
            db = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1).fit(X_feats)
            dbscan_labels = db.labels_
            n_clusters_found = len(set(dbscan_labels) - {-1})
            
            if n_clusters_found < 2:
                print(f"Warning: DBSCAN found only {n_clusters_found} clusters. Skipping this period.")
                continue
            
            # Update classifier head to match number of clusters found
            model.update_head(n_clusters_found, feature_dim)
            # Reinitialize optimizer with new model parameters
            optimizer = optimizer_class(model.parameters(), lr=args.dc_optimizer.lr)
            
            # Train classifier on DBSCAN labels
            model.train()
            for epoch in range(args.brb.reset_interval):
                epoch_loss = 0.0
                n_batches = 0
                
                for batch_idx, batch in enumerate(dc_train_dl):
                    # Dataloader returns (index, data, labels)
                    # batch[0] = indices, batch[1] = data, batch[2] = labels (if provided)
                    batch_data = batch[1].to(device)
                    batch_indices = batch[0].numpy()
                    
                    # Get corresponding DBSCAN labels for this batch using indices
                    batch_dbscan_labels = torch.tensor(dbscan_labels[batch_indices], dtype=torch.long).to(device)
                    
                    # Only train on non-noise points
                    mask = batch_dbscan_labels != -1
                    if mask.any():
                        logits, _ = model(batch_data)
                        loss = criterion(logits[mask], batch_dbscan_labels[mask])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        n_batches += 1
                
                # Log metrics periodically
                if epoch % args.experiment.cluster_log_interval == 0 and run is not None and Y_true is not None:
                    acc_pure = cluster_acc(Y_true, dbscan_labels) # Acc on clustered points only
                    acc_all = cluster_acc_all(Y_true, dbscan_labels) # Acc on entire dataset
                    nmi = nmi_score(Y_true, dbscan_labels)
                    ari = ari_score(Y_true, dbscan_labels)
                    
                    n_noise = np.sum(dbscan_labels == -1)
                    coverage = (len(dbscan_labels) - n_noise) / len(dbscan_labels)

                    log_dict = {
                        f"Training/Period": period + 1,
                        f"Training/Epoch": epoch,
                        f"Training/Loss": epoch_loss / max(n_batches, 1),
                        f"Training/ACC_pure": acc_pure,
                        f"Training/ACC_all": acc_all,
                        f"Training/NMI": nmi,
                        f"Training/ARI": ari,
                        f"Training/Coverage": coverage,
                        f"Training/N_Clusters": n_clusters_found,
                        f"Training/N_Noise": int(n_noise),
                    }
                    run.log(log_dict)
            
            # Evaluate at end of period
            if Y_true is not None:
                acc_all = cluster_acc_all(Y_true, dbscan_labels)
                nmi = nmi_score(Y_true, dbscan_labels)
                n_noise = np.sum(dbscan_labels == -1)
                print(f">> Period {period+1} Result | ACC-All: {acc_all:.4f} | NMI: {nmi:.4f} | K: {n_clusters_found} | Noise: {n_noise} pts")
            else:
                print(f">> Period {period+1} Result | Found K: {n_clusters_found}")
            metrics_log.append({
                'period': period + 1,
                'acc_pure': acc_pure if Y_true is not None else 0.0,
                'acc_all': acc_all if Y_true is not None else 0.0,
                'n_clusters': n_clusters_found,
                'n_noise': np.sum(dbscan_labels == -1)
            })
        
        # Final evaluation on test set
        model.eval()
        test_embeddings = encode_batchwise(test_dl, ae, device)
        db_test = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples, n_jobs=-1).fit(test_embeddings)
        
        # For evaluation, we need cluster centers - use centroids of DBSCAN clusters
        cluster_centers = []
        for cluster_id in set(db_test.labels_) - {-1}:
            cluster_points = test_embeddings[db_test.labels_ == cluster_id]
            if len(cluster_points) > 0:
                cluster_centers.append(cluster_points.mean(axis=0))
        cluster_centers = np.array(cluster_centers) if cluster_centers else np.zeros((1, test_embeddings.shape[1]))
        
        n_test_clusters = len(cluster_centers)
        
        # Only evaluate if we have at least 2 clusters (needed for uncertainty score and other metrics)
        if n_test_clusters >= 2:
            # Evaluate final performance
            metrics, _ = evaluate_deep_clustering(
                cluster_centers=cluster_centers,
                model=ae.to(device),
                dataloader=test_dl,
                labels=test_labels,
                old_labels=None,
                loss_fn=torch.nn.MSELoss(),
                metrics_dict=None,
                return_labels=False,
                track_silhouette=True,
                track_purity=True,
                device=device,
                track_voronoi=args.experiment.track_voronoi,
                track_uncertainty_plot=args.experiment.track_uncertainty_plot,
            )
            
            # Log final scores on test set
            if run is not None:
                metric_dict = {f"Clustering test metrics/{k}": v[-1] for k, v in metrics.items() if "Change" not in k}
                metric_dict["Clustering epoch"] = args.clustering_epochs
                run.log(metric_dict)
        else:
            print(f"Warning: DBSCAN found only {n_test_clusters} cluster(s) on test set. Skipping evaluation (needs at least 2 clusters).")
            print(f"Try adjusting --dbscan-eps (currently {args.dbscan_eps}) or --dbscan-min-samples (currently {args.dbscan_min_samples})")
        
        ae.to("cpu")
        
    else:
        print("Deep Clustering is skipped")
    
    if args.save_clustering_model:
        brb = "baseline"
        if args.brb.reset_weights:
            brb = "brb"
        
        torch.save(
            {
                "sd": ae.state_dict(),
                "dbscan_labels": dbscan_labels if args.clustering_epochs > 0 else None,
                "cluster_centers": cluster_centers if args.clustering_epochs > 0 else None,
            },
            f"{data_path}/clustering_models/dbscan_{args.dataset_name}_{brb}_{args.seed}.pth",
        )
    wandb.finish()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train_dbscan(args)
