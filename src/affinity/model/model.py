import gc
from typing import Any, Optional

import numpy as np
import torch
import gc  
import torch._dynamo
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule
from torch import Tensor, nn
from torchmetrics import MeanMetric
import torch.distributed as dist
from .ema import EMA
from .egnn import ComplexEncoder
from affinity.utils.metrics import calculate_metrics_per_target, calculate_global_auroc, calculate_global_auprc, calculate_value_metrics_per_target
from torch.optim.lr_scheduler import (
    LinearLR,
    LambdaLR,
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau
)

def evaluate_binary_dataset(dataset_idx, dataset_data):
    """Evaluate a single binary classification dataset."""
    scores_data_prob = {}
    for record_id, label, prob in zip(
        dataset_data["record_ids"], 
        dataset_data["labels"], 
        dataset_data["pred_probs"]
    ):
        scores_data_prob[record_id] = {
            "label": label,
            "score": prob,
        }
    
    avg_metrics, _ = calculate_metrics_per_target(scores_data_prob)
    global_auroc = calculate_global_auroc(scores_data_prob)
    
    dataset_metrics = {
        "dataset_idx": dataset_idx,
        "AP": avg_metrics.get("AP", float('nan')),
        "AUROC": avg_metrics.get("AUROC", float('nan')),
        "Global_AUROC": global_auroc,
        "EF_0_5": avg_metrics.get("EF_0_5", float('nan')),
        "EF_1": avg_metrics.get("EF_1", float('nan')),
        "EF_2": avg_metrics.get("EF_2", float('nan')),
        "EF_5": avg_metrics.get("EF_5", float('nan')),
        "length": len(dataset_data["record_ids"])
    }
    
    print(f"--------------------------------")
    print(f"Dataset {dataset_idx} (Binary):")
    print(f"length: {dataset_metrics['length']}")
    print(f"AP: {dataset_metrics['AP']}")
    print(f"AUROC: {dataset_metrics['AUROC']}")
    print(f"EF_0_5: {dataset_metrics['EF_0_5']}")
    print(f"EF_1: {dataset_metrics['EF_1']}")
    print(f"EF_2: {dataset_metrics['EF_2']}")
    print(f"EF_5: {dataset_metrics['EF_5']}")
    print(f"Global AUROC: {dataset_metrics['Global_AUROC']}")
    print(f"--------------------------------")
    
    return dataset_metrics


def evaluate_value_dataset(dataset_idx, dataset_data):
    """Evaluate a single value regression dataset."""
    scores_data_value = {}
    for record_id, label, pred_value in zip(
        dataset_data["record_ids"], 
        dataset_data["labels"], 
        dataset_data["pred_values"]
    ):
        scores_data_value[record_id] = {
            "label": label,
            "score": pred_value,
        }
    
    avg_metrics, _ = calculate_value_metrics_per_target(scores_data_value)
    
    dataset_metrics = {
        "dataset_idx": dataset_idx,
        "Pearson_R": avg_metrics.get("Pearson_R", float('nan')),
        "Kendall_Tau": avg_metrics.get("Kendall_Tau", float('nan')),
        "PMAE": avg_metrics.get("PMAE", float('nan')),
        "MAE": avg_metrics.get("MAE", float('nan')),
        "MAE_cent": avg_metrics.get("MAE_cent", float('nan')),
        "Perc_1kcal": avg_metrics.get("Perc_1kcal", float('nan')),
        "Perc_2kcal": avg_metrics.get("Perc_2kcal", float('nan')),
        "Perc_1kcal_cent": avg_metrics.get("Perc_1kcal_cent", float('nan')),
        "Perc_2kcal_cent": avg_metrics.get("Perc_2kcal_cent", float('nan')),
        "length": len(dataset_data["record_ids"])
    }
    
    print(f"--------------------------------")
    print(f"Dataset {dataset_idx} (Value):")
    print(f"length: {dataset_metrics['length']}")
    print(f"Pearson R: {dataset_metrics['Pearson_R']:.4f}")
    print(f"Kendall Tau: {dataset_metrics['Kendall_Tau']:.4f}")
    print(f"PMAE: {dataset_metrics['PMAE']:.4f}")
    print(f"MAE: {dataset_metrics['MAE']:.4f}")
    print(f"MAE (cent): {dataset_metrics['MAE_cent']:.4f}")
    print(f"Perc 1kcal: {dataset_metrics['Perc_1kcal']:.2f}%")
    print(f"Perc 2kcal: {dataset_metrics['Perc_2kcal']:.2f}%")
    print(f"Perc 1kcal (cent): {dataset_metrics['Perc_1kcal_cent']:.2f}%")
    print(f"Perc 2kcal (cent): {dataset_metrics['Perc_2kcal_cent']:.2f}%")
    print(f"--------------------------------")
    
    return dataset_metrics


def evaluate_enzyme_dataset(dataset_idx, dataset_data):
    """Evaluate a single enzyme dataset (only global AUROC and AUPRC)."""
    scores_data_prob = {}
    for record_id, label, prob in zip(
        dataset_data["record_ids"], 
        dataset_data["labels"], 
        dataset_data["pred_probs"]
    ):
        scores_data_prob[record_id] = {
            "label": label,
            "score": prob,
        }

    # Only calculate global AUROC and AUPRC for enzyme datasets
    global_auroc = calculate_global_auroc(scores_data_prob)
    global_auprc = calculate_global_auprc(scores_data_prob)
    
    dataset_metrics = {
        "dataset_idx": dataset_idx,
        "Global_AUROC": global_auroc,
        "Global_AUPRC": global_auprc,
        "length": len(dataset_data["record_ids"])
    }
    
    print(f"--------------------------------")
    print(f"Dataset {dataset_idx} (Enzyme):")
    print(f"length: {dataset_metrics['length']}")
    print(f"Global_AUROC: {dataset_metrics['Global_AUROC']}")
    print(f"Global_AUPRC: {dataset_metrics['Global_AUPRC']}")
    print(f"--------------------------------")
    
    return dataset_metrics


def focal_loss(pred_logits: Tensor, label: Tensor, alpha: float = 0.25, gamma: float = 2.0) -> Tensor:
    """
    Compute focal loss for binary classification using logits.
    
    Parameters
    ----------
    pred_logits : Tensor
        Predicted logits (before sigmoid) [B]
    label : Tensor
        Ground truth labels [B]
    alpha : float
        Weighting factor for rare class
    gamma : float
        Focusing parameter
        
    Returns
    -------
    Tensor
        Focal loss value
    """
    # Calculate focal loss using logits for BCE
    ce_loss = F.binary_cross_entropy_with_logits(pred_logits, label, reduction='none')
    p_t = torch.exp(-ce_loss)
    focal_weight = alpha * label + (1 - alpha) * (1 - label)
    focal_weight = focal_weight * (1 - p_t).clamp(min=1e-7) ** gamma
    
    focal_loss = focal_weight * ce_loss
    return focal_loss.mean()

def abs_huber_loss(pred_value: Tensor, label: Tensor, delta: float = 0.5) -> Tensor:
    """
    Compute absolute loss for regression using predicted values.
    """
    # Mask for different qualifiers
    
    return F.huber_loss(pred_value, label, reduction='none', delta=delta).mean()

def dif_huber_loss(pred_value: Tensor, label: Tensor, delta: float = 0.5, group_size: int = 5) -> Tensor:
    """
    Compute difference loss for regression using predicted values within groups.
    Each group has size `group_size = 5`, and we compute pairwise differences within each group.
    """
    batch_size = pred_value.shape[0]
    
    assert batch_size % group_size == 0, f"batch size {batch_size} must be divisible by group size {group_size}"
    assert batch_size >= group_size, f"batch size {batch_size} must be greater than or equal to group size {group_size}"
    
    num_groups = batch_size // group_size
    
    # Reshape to (num_groups, group_size, ...)
    original_shape = pred_value.shape[1:]
    pred_reshaped = pred_value.view(num_groups, group_size, *original_shape)
    label_reshaped = label.view(num_groups, group_size, *original_shape)
    
    # Compute differences within each group (adjacent pairs)
    pred_diff = pred_reshaped[:, 1:] - pred_reshaped[:, :-1]  # shape: (num_groups, group_size - 1, ...)
    label_diff = label_reshaped[:, 1:] - label_reshaped[:, :-1]
    
    # Flatten back for huber loss computation
    pred_diff_flat = pred_diff.reshape(-1, *original_shape)
    label_diff_flat = label_diff.reshape(-1, *original_shape)
    
    return F.huber_loss(pred_diff_flat, label_diff_flat, reduction='none', delta=delta).mean()

def apply_correction(pred_value: float, mw: Optional[float], C0: float = 2.177929, C1: float = -0.218761, C2: float = 1.472271) -> float:
    """Apply correction to pred_value using molecular weight.
    
    Corrected value = C0 * pred_value + C1 * mw^0.3 + C2
    
    Parameters
    ----------
    pred_value : float
        Original predicted value
    mw : float
        Molecular weight
    C0 : float, optional
        Correction parameter (default: 2.177929)
    C1 : float, optional
        Correction parameter (default: -0.218761)
    C2 : float, optional
        Correction parameter (default: 1.472271)
    
    Returns
    -------
    float
        Corrected predicted value
    """
    if mw is None:
        return pred_value
    return C0 * pred_value + C1 * (mw ** 0.3) + C2

class AffinityModel(LightningModule):
    def __init__(
        self,
        training_args: dict[str, Any],
        encoder_args: dict[str, Any],
        predictor_args: dict[str, Any],
        loss_args: Optional[dict[str, Any]] = None,
        compile_encoder: bool = False,
        ema: bool = False,
        ema_decay: float = 0.999,
        use_morgan: bool = False,
        use_unimol: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # EMA setup
        self.use_ema = self.hparams.ema
        self.ema_decay = self.hparams.ema_decay

        # Args
        self.training_args = self.hparams.training_args

        # Loss configuration
        self.loss_args = self.hparams.loss_args if self.hparams.loss_args is not None else {}

        # Loss weights
        self.value_loss_weight = self.hparams.training_args.get("value_loss_weight", 1.0)
        self.prob_loss_weight = self.hparams.training_args.get("prob_loss_weight", 1.0)
        self.enzyme_loss_weight = self.hparams.training_args.get("enzyme_loss_weight", 1.0)

        if "alpha" not in self.loss_args or "gamma" not in self.loss_args:
            self.loss_args["alpha"] = 0.6
            self.loss_args["gamma"] = 1.0
            print(f"Warning: Focal loss requires 'alpha' and 'gamma' parameters in loss_args, defaulting to {self.loss_args['alpha']} and {self.loss_args['gamma']}")

        if "delta" not in self.loss_args or "group_size" not in self.loss_args:
            self.loss_args["delta"] = 0.5
            self.loss_args["group_size"] = 5
            print(f"Warning: HuberLoss requires 'delta' and 'group_size' parameters in loss_args, defaulting to {self.loss_args['delta']} and {self.loss_args['group_size']}")

        # Metrics
        self.train_affinity_loss_logger = MeanMetric()
        self.train_value_loss_logger = MeanMetric()
        self.train_abs_loss_logger = MeanMetric()
        self.train_dif_loss_logger = MeanMetric()
        self.train_prob_loss_logger = MeanMetric()
        self.train_enzyme_loss_logger = MeanMetric()

        # Dynamo
        torch._dynamo.config.cache_size_limit = 512
        torch._dynamo.config.accumulated_cache_size_limit = 512

        # Encoder module
        self.encoder = ComplexEncoder(**self.hparams.encoder_args)
        # self.encoder = PairformerComplexEncoder(**self.hparams.encoder_args)

        # Morgan feature processor (optional)
        self.use_morgan = self.hparams.use_morgan
        if self.use_morgan:
            self.morgan_mlp = nn.Sequential(
                nn.Linear(2048, self.hparams.predictor_args["hidden_dim"] * 2),  # 2048 -> 192
                nn.SiLU(),
                nn.Linear(self.hparams.predictor_args["hidden_dim"] * 2, self.hparams.predictor_args["hidden_dim"]),  # 192 -> 192
            )

        # UniMol feature processor (optional)
        self.use_unimol = self.hparams.use_unimol
        if self.use_unimol:
            self.unimol_mlp = nn.Sequential(
                nn.Linear(768, self.hparams.predictor_args["hidden_dim"] * 2),  # 768 -> 192
                nn.SiLU(),
                nn.Linear(self.hparams.predictor_args["hidden_dim"] * 2, self.hparams.predictor_args["hidden_dim"]),  # 192 -> 192
            )

        # Calculate predictor input dimension
        base_dim = self.hparams.encoder_args["output_dim"]  # 192

        # MLP predictors
        self.value_predictor = nn.Sequential(
            nn.Linear(base_dim, self.hparams.predictor_args["hidden_dim"]),
            nn.SiLU(),
            nn.Linear(self.hparams.predictor_args["hidden_dim"], self.hparams.predictor_args["hidden_dim"]), 
            nn.SiLU(),
            nn.Linear(self.hparams.predictor_args["hidden_dim"], 1),
        )

        self.prob_predictor = nn.Sequential(
            nn.Linear(base_dim, self.hparams.predictor_args["hidden_dim"]),
            nn.SiLU(),
            nn.Linear(self.hparams.predictor_args["hidden_dim"], self.hparams.predictor_args["hidden_dim"]), 
            nn.SiLU(),
            nn.Linear(self.hparams.predictor_args["hidden_dim"], 1),
            nn.Linear(1, 1)
        )

        predictor_input_dim = base_dim + \
                             (self.hparams.predictor_args["hidden_dim"] if self.use_morgan else 0) + \
                             (self.hparams.predictor_args["hidden_dim"] if self.use_unimol else 0)  # 192, 384, 576, or 768

        # Enzyme predictor - separate MLP head for enzyme predictions
        self.enzyme_predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, self.hparams.predictor_args["hidden_dim"]),
            nn.SiLU(),
            nn.Linear(self.hparams.predictor_args["hidden_dim"], self.hparams.predictor_args["hidden_dim"]), 
            nn.SiLU(),
            nn.Linear(self.hparams.predictor_args["hidden_dim"], 1),
            nn.Linear(1, 1)
        )

        # Compile modules if requested
        self.is_encoder_compiled = self.hparams.compile_encoder

        if self.hparams.compile_encoder:
            self.encoder = torch.compile(self.encoder, dynamic=False, fullgraph=False)

        # Compile affinity module
        self.validation_step_outputs = []

    def forward(
        self,
        feats: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Forward pass.
        
        Parameters
        ----------
        feats : dict[str, Tensor]
            Input features

        Returns
        -------
        dict[str, Tensor]
            Dictionary containing affinity predictions
        """

        if getattr(self, "is_encoder_compiled", False) and not self.training and hasattr(self.encoder, "_orig_mod"):
            encoder = self.encoder._orig_mod
        else:
            encoder = self.encoder

        complex_edge_repr = torch.stack([
            feats["edge_index"][:, 0, :],  # node_in
            feats["edge_index"][:, 1, :],  # node_out  
            feats["edge_types"]            # edge_type
        ], dim=-1)  # [B, E, 3]
    
        atom_types = F.one_hot((feats["atom_types"] + 1).clamp(0, 23), num_classes=24).float()  # [B, L, 24] # +1 because -1 is global node feature
        residue_types = F.one_hot((feats["residue_types"] + 1).clamp(0, 23), num_classes=24).float()  # [B, L, 24]

        # One-hot encode molecule_types to 3 channels (padding=0, protein=1, ligand=2)
        molecule_types = F.one_hot(feats["molecule_types"].clamp(0, 2), num_classes=3).float()  # [B, L, 3]

        # Scale residue_indices to [0, 1] per sample to balance feature magnitudes
        residue_indices = feats["residue_indices"].float()  # [B, L]
        mask_valid = residue_indices >= 0                  # [B, L] valid if >= 0
        den = residue_indices.masked_fill(~mask_valid, 0).amax(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
        residue_indices = torch.where(mask_valid, residue_indices / den, residue_indices)  # keep -1 untouched

        # Process complex with encoder
        complex_repr = encoder(
            complex_coord=feats["coords"],
            protein_repr=feats["protein_repr"],
            ligand_repr=feats["ligand_repr"],
            complex_edge_repr=complex_edge_repr,
            complex_edge_mask=feats["edge_mask"],
            molecule_types=molecule_types,
            atom_types=atom_types,
            residue_types=residue_types,
            residue_indices=residue_indices,
            complex_mask=feats["atom_mask"] ,
        )

        # Global pooling for complex-level prediction
        complex_mask_expanded = feats["atom_mask"].unsqueeze(-1)  # [B, L, 1]
        pooled_repr = (complex_repr * complex_mask_expanded).sum(dim=1) / (complex_mask_expanded.sum(dim=1) + 1e-8)  # [B, output_dim]

        # Collect all features to concatenate
        features_to_concat = [pooled_repr]  # Start with EGNN output [B, 192]

        # Process Morgan features if available and model is configured to use them
        if self.use_morgan:
            # Ensure Morgan features are available when model expects them
            assert "morgan_repr" in feats, "Model expects morgan_repr but not found in batch"
            # Ensure Morgan features are on the same device as the model
            morgan_repr = feats["morgan_repr"].to(pooled_repr.device)
            morgan_features = self.morgan_mlp(morgan_repr)  # [B, 2048] -> [B, 192]
            features_to_concat.append(morgan_features)

        # Process UniMol features if available and model is configured to use them
        if self.use_unimol:
            # Ensure UniMol features are available when model expects them
            assert "unimol_repr" in feats, "Model expects unimol_repr but not found in batch"
            # Ensure UniMol features are on the same device as the model
            unimol_repr = feats["unimol_repr"].to(pooled_repr.device)
            unimol_features = self.unimol_mlp(unimol_repr)  # [B, 768] -> [B, 192]
            features_to_concat.append(unimol_features)

        # Concatenate all features
        final_repr = torch.cat(features_to_concat, dim=-1)  # [B, 192/384/576/768]

        # Predict affinity value and binding probability
        affinity_value = self.value_predictor(pooled_repr)  # [B, 1]
        affinity_logits = self.prob_predictor(pooled_repr)  # [B, 1]
        enzyme_logits = self.enzyme_predictor(final_repr)  # [B, 1]

        return {
            "affinity_pred_value": affinity_value,
            "affinity_logits_binary": affinity_logits,
            "enzyme_logits": enzyme_logits,
        }

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Training step."""

        try:
            # Execute forward propagation
            out = self(feats=batch)

            # Calculate loss
            loss_dict = self.compute_loss(out, batch)

        except Exception as e:
            print(f"Skipping batch {batch_idx}, error: {e}")
            import traceback
            traceback.print_exc()
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict = {"total_loss": loss, "value_loss": loss, "abs_loss": loss, "dif_loss": loss, "prob_loss": loss}

        # Always log the loss for ModelCheckpoint monitoring    
        self.train_affinity_loss_logger.update(loss_dict["total_loss"])
        self.log("train/affinity_loss", loss_dict["total_loss"], on_step=True, on_epoch=False, prog_bar=False)
        if loss_dict["value_loss"] > 0:
            self.train_value_loss_logger.update(loss_dict["value_loss"])
            self.train_abs_loss_logger.update(loss_dict["abs_loss"])
            self.train_dif_loss_logger.update(loss_dict["dif_loss"])
            self.log("train/value_loss", loss_dict["value_loss"], on_step=True, on_epoch=False, prog_bar=False)
            self.log("train/abs_loss", loss_dict["abs_loss"], on_step=True, on_epoch=False, prog_bar=False)
            self.log("train/dif_loss", loss_dict["dif_loss"], on_step=True, on_epoch=False, prog_bar=False)
        if loss_dict["prob_loss"] > 0:
            self.train_prob_loss_logger.update(loss_dict["prob_loss"]) 
            self.log("train/prob_loss", loss_dict["prob_loss"], on_step=True, on_epoch=False, prog_bar=False)
        if loss_dict["enzyme_loss"] > 0:
            self.train_enzyme_loss_logger.update(loss_dict["enzyme_loss"]) 
            self.log("train/enzyme_loss", loss_dict["enzyme_loss"], on_step=True, on_epoch=False, prog_bar=False)
        self.training_log()

        # Dynamically clear CUDA cache when reserved memory significantly exceeds allocated memory
        if torch.cuda.is_available():
            try:
                reserved = torch.cuda.memory_reserved()
                allocated = torch.cuda.memory_allocated()
                threshold_bytes = int(getattr(self, "cuda_cache_fragment_threshold_bytes", 3.5 * 1024 ** 3))
                if reserved - allocated > threshold_bytes:
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
            except Exception:
                pass

        return loss_dict["total_loss"]

    def compute_loss(self, out: dict[str, Tensor], batch: dict[str, Tensor]) -> dict:
        """Compute affinity loss.
        
        Parameters
        ----------
        out : dict[str, Tensor]
            Model output containing affinity predictions
        batch : dict[str, Tensor]
            Batch containing labels
            
        Returns
        -------
        dict[str, Tensor]
            Dictionary containing total loss and individual loss components
        """
        # Extract predictions and labels
        pred_value = out["affinity_pred_value"].squeeze(-1)  # [B]
        pred_logits = out["affinity_logits_binary"].squeeze(-1)  # [B]
        pred_enzyme_logits = out["enzyme_logits"].squeeze(-1)
        
        abs_loss = 0.0
        dif_loss = 0.0
        value_loss = 0.0
        prob_loss = 0.0
        enzyme_loss = 0.0

        if "label" in batch and "label_type" in batch:
            labels = batch["label"]  # [B]
            label_types = batch["label_type"]  # [B]
            
            # Separate binary, enzyme, and value samples
            binary_mask = (label_types == 1)
            value_mask = (label_types == 2)
            enzyme_mask = (label_types == 3)
            
            # Process binary samples
            if binary_mask.any():
                binary_labels = labels[binary_mask]
                binary_logits = pred_logits[binary_mask]
                binary_prob = torch.sigmoid(binary_logits)
                
                prob_loss = focal_loss(binary_logits, binary_labels, 
                                    alpha=self.loss_args["alpha"], 
                                    gamma=self.loss_args["gamma"])
                
            # Process enzyme samples (use focal loss)
            if enzyme_mask.any():
                enzyme_labels = labels[enzyme_mask]
                enzyme_logits = pred_enzyme_logits[enzyme_mask]  # Use dedicated enzyme head
                enzyme_prob = torch.sigmoid(enzyme_logits)
                # Use focal loss for enzyme samples
                enzyme_loss = focal_loss(enzyme_logits, enzyme_labels, 
                                       alpha=self.loss_args["alpha"], 
                                       gamma=self.loss_args["gamma"])

            # Process value samples  
            if value_mask.any():
                value_labels = labels[value_mask]
                value_preds = pred_value[value_mask]
                
                # For value loss, we can only compute abs_loss for mixed batches
                # since dif_loss requires consecutive groups
                abs_loss = abs_huber_loss(value_preds, value_labels, delta=self.loss_args["delta"])
                
                # Only compute dif_loss if we have enough samples and they form complete groups
                dif_loss = dif_huber_loss(value_preds, value_labels, 
                                        delta=self.loss_args["delta"], 
                                        group_size=self.loss_args["group_size"])
                value_loss = 0.1 * abs_loss + 0.9 * dif_loss
                    
        total_loss = (self.prob_loss_weight * prob_loss + 
                     self.value_loss_weight * value_loss +
                     self.enzyme_loss_weight * enzyme_loss)

        return {
            "total_loss": total_loss,
            "value_loss": value_loss,
            "abs_loss": abs_loss,
            "dif_loss": dif_loss,
            "prob_loss": prob_loss,
            "enzyme_loss": enzyme_loss,
        }

    def training_log(self):
        """Training log."""
        # Note: grad_norm is logged in optimizer_step after clipping
        self.log("train/param_norm", self.parameter_norm(self), prog_bar=False)

        # Log learning rates for different groups
        for i, group in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"lr_group_{i}", group['lr'], prog_bar=False)

        self.log(
            "train/param_norm_encoder",
            self.parameter_norm(self.encoder),
            prog_bar=False,
        )

        self.log(
            "train/param_norm_predictor",
            self.parameter_norm(nn.ModuleList([self.value_predictor, self.prob_predictor, self.enzyme_predictor])),
            prog_bar=False,
        )

    def on_train_epoch_end(self):
        """Training epoch end."""
        self.log(
            "train/affinity_loss",
            self.train_affinity_loss_logger,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/value_loss_epoch",
            self.train_value_loss_logger,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/abs_loss_epoch",
            self.train_abs_loss_logger,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/dif_loss_epoch",
            self.train_dif_loss_logger,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/prob_loss_epoch",
            self.train_prob_loss_logger,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/enzyme_loss_epoch",
            self.train_enzyme_loss_logger,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

    def gradient_norm(self, module):
        """Compute gradient norm."""
        parameters = [
            p.grad.norm(p=2) ** 2
            for p in module.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if len(parameters) == 0:
            return torch.tensor(
                0.0, device="cuda" if torch.cuda.is_available() else "cpu"
            )
        norm = torch.stack(parameters).sum().sqrt()
        return norm

    def parameter_norm(self, module):
        """Compute parameter norm."""
        parameters = [p.norm(p=2) ** 2 for p in module.parameters() if p.requires_grad]
        if len(parameters) == 0:
            return torch.tensor(
                0.0, device="cuda" if torch.cuda.is_available() else "cpu"
            )
        norm = torch.stack(parameters).sum().sqrt()
        return norm

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """Predict step - EMA handled by Callback when enabled."""
        try:
            out = self(feats=batch)
            
            pred_dict = {"exception": False}

            # Add batch information
            pred_dict["record_id"] = batch["record_id"].cpu()

            # Add output information
            pred_dict["affinity_pred_value_raw"] = out["affinity_pred_value"].cpu()
            pred_dict["affinity_pred_value"] = apply_correction(out["affinity_pred_value"].cpu(), batch["mw"].cpu())
            pred_dict["affinity_probability_binary"] = torch.sigmoid(out["affinity_logits_binary"].cpu())
            pred_dict["affinity_probability_enzyme"] = torch.sigmoid(out["enzyme_logits"].cpu())
            pred_dict["mw"] = batch["mw"].cpu()
            return pred_dict

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| Warning: out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return {"exception": True}
            else:
                raise e

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict:
        
        result = {}
        try:
            out = self(feats=batch)
            result = {
                "pred_value": out["affinity_pred_value"].detach(),
                "pred_prob": torch.sigmoid(out["affinity_logits_binary"]).detach(),
                "enzyme_prob": torch.sigmoid(out["enzyme_logits"]).detach(),
                "label": batch["label"].detach().float(),
                "record_id": batch["record_id"].detach(),
                "type": batch["label_type"].detach(),
                "dataset_idx": batch["dataset_idx"].detach(),
            }
            self.validation_step_outputs.append(result)
        except Exception as e:
            print(f"Validation batch {batch_idx} error: {e}")
        
        return result

    def on_validation_epoch_end(self):
        """Validation epoch end."""
        
        outputs = self.all_gather(self.validation_step_outputs)
        if self.trainer.global_rank == 0:
            # First, deduplicate data based on record_id
            unique_samples = {}
            
            for output in outputs:
                # Extract data from each output
                dataset_indices = output["dataset_idx"].flatten().tolist()
                pred_values = output["pred_value"].flatten().tolist()
                pred_probs = output["pred_prob"].flatten().tolist()
                enzyme_probs = output["enzyme_prob"].flatten().tolist()
                labels = output["label"].flatten().tolist()
                types = output["type"].flatten().tolist()
                
                # Parse record IDs
                record_ids_tensor = output["record_id"].reshape(-1, output["record_id"].shape[-1]).cpu().numpy()
                record_ids = [''.join(chr(x) for x in row if x != 0).rstrip('-') for row in record_ids_tensor]

                # Store unique samples using (dataset_idx, record_id) as key
                for dataset_idx, pred_value, pred_prob, enzyme_prob, label, data_type, record_id in zip(
                    dataset_indices, pred_values, pred_probs, enzyme_probs, labels, types, record_ids
                ):
                    unique_key = (dataset_idx, record_id)
                    if unique_key not in unique_samples:
                        unique_samples[unique_key] = {
                            'pred_value': pred_value,
                            'pred_prob': pred_prob,
                            'enzyme_prob': enzyme_prob,
                            'label': label,
                            'type': data_type,
                            'dataset_idx': dataset_idx,
                            'record_id': record_id
                        }
            
            # Group deduplicated data by dataset_idx
            dataset_groups = {}
            for sample in unique_samples.values():
                dataset_idx = sample['dataset_idx']
                if dataset_idx not in dataset_groups:
                    dataset_groups[dataset_idx] = {
                        "type": sample['type'],
                        "pred_values": [],
                        "pred_probs": [],
                        "labels": [],
                        "record_ids": []
                    }
                
                dataset_groups[dataset_idx]["pred_values"].append(sample['pred_value'])
                # Use enzyme_prob for enzyme datasets, pred_prob for others
                if sample['type'] == 3:  # enzyme type
                    dataset_groups[dataset_idx]["pred_probs"].append(sample['enzyme_prob'])
                else:
                    dataset_groups[dataset_idx]["pred_probs"].append(sample['pred_prob'])
                dataset_groups[dataset_idx]["labels"].append(sample['label'])
                dataset_groups[dataset_idx]["record_ids"].append(sample['record_id'])

            # Evaluate each dataset separately
            binary_tasks = [(idx, data) for idx, data in dataset_groups.items() if data["type"] == 1]
            value_tasks = [(idx, data) for idx, data in dataset_groups.items() if data["type"] == 2]
            enzyme_tasks = [(idx, data) for idx, data in dataset_groups.items() if data["type"] == 3]
             
            all_binary_metrics = []
            all_value_metrics = []
            all_enzyme_metrics = []
            
            # Use sequential processing to avoid NCCL communication conflicts in distributed training
            if binary_tasks:
                all_binary_metrics = [
                    evaluate_binary_dataset(idx, data) for idx, data in binary_tasks
                ]
            if value_tasks:
                all_value_metrics = [
                    evaluate_value_dataset(idx, data) for idx, data in value_tasks
                ]
            if enzyme_tasks:
                all_enzyme_metrics = [
                    evaluate_enzyme_dataset(idx, data) for idx, data in enzyme_tasks
                ]
                
                # Merge all enzyme datasets and evaluate together
                merged_enzyme_data = {
                    "record_ids": [],
                    "labels": [],
                    "pred_probs": []
                }
                
                for dataset_idx, data in enzyme_tasks:
                    # Add dataset prefix to record_ids to ensure uniqueness
                    prefixed_ids = [f"{dataset_idx}-{rid}" for rid in data["record_ids"]]
                    merged_enzyme_data["record_ids"].extend(prefixed_ids)
                    merged_enzyme_data["labels"].extend(data["labels"])
                    merged_enzyme_data["pred_probs"].extend(data["pred_probs"])
                
                # Evaluate merged enzyme dataset
                merged_enzyme_metrics = evaluate_enzyme_dataset("enzyme_all", merged_enzyme_data)
                all_enzyme_metrics.append(merged_enzyme_metrics)

            # Log metrics for each dataset separately
            _val_metrics = {}
            
            # Log binary dataset metrics
            for binary_metrics in all_binary_metrics:
                dataset_idx = binary_metrics["dataset_idx"]
                for metric_name in ["AP", "AUROC", "Global_AUROC", "EF_0_5", "EF_1", "EF_2", "EF_5"]:
                    _val_metrics[f"val/{dataset_idx}_{metric_name}"] = binary_metrics[metric_name]
            
            # Log value dataset metrics
            for value_metrics in all_value_metrics:
                dataset_idx = value_metrics["dataset_idx"]
                metric_names = ["Pearson_R", "Kendall_Tau", "PMAE", "MAE", "MAE_cent", 
                               "Perc_1kcal", "Perc_2kcal", "Perc_1kcal_cent", "Perc_2kcal_cent"]
                for metric_name in metric_names:
                    _val_metrics[f"val/{dataset_idx}_{metric_name}"] = value_metrics[metric_name]
            
            # Log enzyme dataset metrics (including merged "enzyme_all")
            for enzyme_metrics in all_enzyme_metrics:
                dataset_idx = enzyme_metrics["dataset_idx"]
                for metric_name in ["Global_AUROC", "Global_AUPRC"]:
                    _val_metrics[f"val/{dataset_idx}_{metric_name}"] = enzyme_metrics[metric_name]
            
        else:
            _val_metrics = {}

        _val_metrics = self.trainer.strategy.broadcast(_val_metrics, src=0)
        self.log_dict(_val_metrics, sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_callbacks(self) -> list[Callback]:
        """Configure model callbacks."""
        if self.use_ema:
            # Keep defaults aligned with boltz
            return [EMA(decay=self.ema_decay, apply_ema_every_n_steps=1, start_step=0, eval_with_ema=True, warm_start=True)]
        return []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer and scheduler.

        Supports two modes controlled by training_args["optimizer_type"]:
        - "adamw" (default): single AdamW over all parameters.
        - "muon": Muon for hidden weights (ndim >= 2) in encoder, AdamW for the rest.
        """

        ta = self.training_args
        scheduler_type = str(ta.get("lr_scheduler", "cosine_restart")).lower()

        base_lr: float = float(ta.get("base_lr", 0.0))
        max_lr: float = float(ta.get("max_lr", 1e-4))
        min_lr: float = float(ta.get("min_lr", base_lr))
        weight_decay: float = float(ta.get("weight_decay", 0.0))
        optimizer_type: str = str(ta.get("optimizer_type", "adamw")).lower()

        if max_lr <= 0.0:
            print("| Warning: max_lr <= 0.0 provided; using 1e-12 to avoid zero LR")
            max_lr = 1e-12
        if min_lr < 0.0:
            min_lr = 0.0
        if min_lr > max_lr:
            print("| Warning: min_lr > max_lr; clamping min_lr to max_lr")
            min_lr = max_lr

        if optimizer_type == "muon":
            try:
                from affinity.utils.muon import MuonWithAuxAdam
            except Exception as e:
                raise ImportError(
                    "Local Muon optimizer not available. Ensure affinity.utils.muon is importable."
                ) from e

            encoder_module = self.encoder._orig_mod if hasattr(self.encoder, "_orig_mod") else self.encoder
            hidden_weights = [
                p for p in encoder_module.parameters() if p.requires_grad and getattr(p, "ndim", 0) >= 2
            ]
            scalar_weights = [
                p for p in encoder_module.parameters() if p.requires_grad and getattr(p, "ndim", 0) < 2
            ]
            predictor_params = list(self.value_predictor.parameters()) + \
                       list(self.prob_predictor.parameters()) + \
                       list(self.enzyme_predictor.parameters())
            
            if self.use_morgan:
                predictor_params += list(self.morgan_mlp.parameters())
            if self.use_unimol:
                predictor_params += list(self.unimol_mlp.parameters())
            
            nonhidden_params = [p for p in scalar_weights + predictor_params if p.requires_grad]
            
            if len(hidden_weights) == 0:
                print("| Warning: Muon selected but no hidden weights (ndim>=2) in encoder; falling back to AdamW.")
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=max_lr,
                    betas=(ta.adam_beta_1, ta.adam_beta_2),
                    eps=ta.adam_eps,
                    weight_decay=weight_decay,
                )
            else:
                muon_lr: float = float(ta.get("muon_lr", 0.002))
                muon_weight_decay: float = float(ta.get("muon_weight_decay", 0.01))
                param_groups = [
                    dict(params=hidden_weights, use_muon=True, lr=muon_lr, weight_decay=muon_weight_decay),
                    dict(
                        params=nonhidden_params,
                        use_muon=False,
                        lr=max_lr,
                        betas=(ta.adam_beta_1, ta.adam_beta_2),
                        eps=ta.adam_eps,
                        weight_decay=weight_decay,
                    ),
                ]
                optimizer = MuonWithAuxAdam(param_groups)
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=max_lr,
                betas=(ta.adam_beta_1, ta.adam_beta_2),
                eps=ta.adam_eps,
                weight_decay=weight_decay,
            )

        # Scheduler selection (warmup is handled in optimizer_step)
        if scheduler_type == "cosine_restart":
            t_0 = int(ta.get("t_0", 80000))
            t_mult = int(ta.get("t_mult", 2))
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t_0,
                T_mult=t_mult,
                eta_min=min_lr,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif scheduler_type == "cosine":
            t_max = int(ta.get("total_steps", 1000000))
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=min_lr,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif scheduler_type == "linear_decay":
            decay_steps = int(ta.get("total_steps", 1000000))
            end_factor = 0.0 if max_lr == 0 else (min_lr / max_lr)
            if end_factor < 0.0:
                end_factor = 0.0
            if end_factor > 1.0:
                end_factor = 1.0
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=end_factor,
                total_iters=decay_steps,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif scheduler_type == "constant":
            scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif scheduler_type == "reduce_on_plateau":
            # ReduceLROnPlateau is epoch-based and requires monitoring a metric
            reduce_lr_factor = float(ta.get("reduce_lr_factor", 0.5))
            reduce_lr_patience = int(ta.get("reduce_lr_patience", 10))
            reduce_lr_monitor = str(ta.get("reduce_lr_monitor", "val/enzyme_all_Global_AUPRC"))
            reduce_lr_mode = str(ta.get("reduce_lr_mode", "max"))
            
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=reduce_lr_mode,
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=min_lr,
            )
            
            return [optimizer], [{
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": reduce_lr_monitor,
                "strict": True,
            }]
        else:
            t_0 = int(ta.get("t_0", 80000))
            t_mult = int(ta.get("t_mult", 2))
            print(f"| Warning: Unknown lr_scheduler '{scheduler_type}', defaulting to cosine_restart")
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t_0,
                T_mult=t_mult,
                eta_min=min_lr,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Custom optimizer step with manual warmup support.
        
        Supports both step-based and epoch-based warmup:
        - warmup_steps > 0: linear warmup based on global step
        - warmup_epochs > 0: linear warmup based on epoch
        Only one should be provided at a time.
        """
        ta = self.training_args
        warmup_steps = int(ta.get("warmup_steps", 0))
        warmup_epochs = int(ta.get("warmup_epochs", 0))
        max_lr = float(ta.get("max_lr", 1e-4))
        base_lr = float(ta.get("base_lr", 0.0))
        muon_lr = float(ta.get("muon_lr", 0.002))
        if warmup_steps > 0 and warmup_epochs > 0:
            print("| Warning: warmup_steps and warmup_epochs are provided; using warmup_steps")
            warmup_epochs = 0
        
        in_warmup = False
        progress = 1.0

        if warmup_steps > 0:
            current_step = self.trainer.global_step
            if current_step <= warmup_steps:
                in_warmup = True
                progress = current_step / warmup_steps
        elif warmup_epochs > 0:
            if epoch <= warmup_epochs:
                in_warmup = True
                progress = epoch / warmup_epochs
    
        if in_warmup:
            for pg in optimizer.param_groups:
                if pg.get('use_muon', False):
                    # Muon param group: base_lr -> muon_lr
                    pg['lr'] = base_lr + (muon_lr - base_lr) * progress
                else:
                    # AdamW param group: base_lr -> max_lr
                    pg['lr'] = base_lr + (max_lr - base_lr) * progress

        # Execute closure first (includes forward + backward)
        if optimizer_closure is not None:
            optimizer_closure()
        
        # Apply gradient clipping after backward
        if self.trainer.gradient_clip_val is not None and self.trainer.gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.trainer.gradient_clip_val,
                gradient_clip_algorithm=self.trainer.gradient_clip_algorithm or "norm",
            )
        
        # Log grad_norm after clipping (only when gradients exist, i.e., after accumulation completes)
        self.log("train/grad_norm", self.gradient_norm(self), prog_bar=False)
        self.log("train/grad_norm_encoder", self.gradient_norm(self.encoder), prog_bar=False)
        self.log("train/grad_norm_predictor", 
                 self.gradient_norm(nn.ModuleList([self.value_predictor, self.prob_predictor, self.enzyme_predictor])), 
                 prog_bar=False)
        
        # Finally step the optimizer
        optimizer.step()
