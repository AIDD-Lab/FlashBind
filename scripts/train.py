"""
Affinity Training Script
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Add the src directory to Python path so we can import affinity
sys.path.insert(0, "./src")

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, listconfig
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from affinity.dataset.training import AffinityTrainingDataModule, AffinityDataConfig


@dataclass
class AffinityTrainConfig:
    """Affinity train configuration.

    Attributes
    ----------
    data : AffinityDataConfig
        The data configuration.
    model : LightningModule
        The model configuration.
    output : str
        The output directory.
    trainer : Optional[dict]
        The trainer configuration.
    resume : Optional[str]
        The resume checkpoint.
    pretrained : Optional[str]
        The pretrained model.
    load_input_embedder : bool
        Load InputEmbedder weights from boltz checkpoint.
    load_affinity_module : bool
        Load AffinityModule weights from boltz checkpoint.
    wandb : Optional[dict]
        The wandb configuration.
    disable_checkpoint : bool
        Disable checkpoint.
    matmul_precision : Optional[str]
        The matmul precision.
    find_unused_parameters : Optional[bool]
        Find unused parameters.
    save_top_k : Optional[int]
        Save top k checkpoints.
    save_every_n_steps : Optional[int]
        Save checkpoint every n steps.
    validation_only : bool
        Run validation only.
    debug : bool
        Debug mode.
    strict_loading : bool
        Fail on mismatched checkpoint weights.
    """

    data: AffinityDataConfig
    model: LightningModule
    output: str
    trainer: Optional[dict] = None
    resume: Optional[str] = None
    pretrained: Optional[str] = None
    load_input_embedder: bool = False
    load_affinity_module: bool = False
    wandb: Optional[dict] = None
    disable_checkpoint: bool = False
    matmul_precision: Optional[str] = None
    find_unused_parameters: Optional[bool] = False
    static_graph: Optional[bool] = True
    save_top_k: Optional[int] = 1
    save_every_n_steps: Optional[int] = 1000
    validation_only: bool = False
    debug: bool = False
    strict_loading: bool = True
    early_stopping: Optional[dict] = None


def train_affinity(raw_config: str, args: list[str]) -> None:
    """Run affinity training.

    Parameters
    ----------
    raw_config : str
        The input yaml configuration.
    args : list[str]
        Any command line overrides.
    """
    # Load the configuration
    raw_config = omegaconf.OmegaConf.load(raw_config)

    # Apply input arguments
    args = omegaconf.OmegaConf.from_dotlist(args)
    raw_config = omegaconf.OmegaConf.merge(raw_config, args)

    # Instantiate the task
    hydra_cfg = hydra.utils.instantiate(raw_config)

    if not hasattr(hydra_cfg.data, 'val_sets'):
        hydra_cfg.data.val_sets = []
    cfg = AffinityTrainConfig(**hydra_cfg)

    print(f"Loss Args: {cfg.model.loss_args}")

    # Set matmul precision
    if cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    # Create output directory if it doesn't exist
    output_dir = Path(cfg.output)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Create trainer dict
    trainer = cfg.trainer
    if trainer is None:
        trainer = {}

    # Flip some arguments in debug mode
    devices = trainer.get("devices", 1)

    wandb = cfg.wandb
    if cfg.debug:
        if isinstance(devices, int):
            devices = 1
        elif isinstance(devices, (list, listconfig.ListConfig)):
            devices = [devices[0]]
        trainer["devices"] = devices
        # Modify num_workers in debug mode
        cfg.data.train_sets.num_workers = 0
        if hasattr(cfg.data, 'val_sets') and cfg.data.val_sets:
            cfg.data.val_sets.num_workers = 0
        if wandb:
            wandb = None

    # Convert OmegaConf config to proper data config and create data module
    data_module = AffinityTrainingDataModule(cfg.data)
    model_module = cfg.model

    print("devices: ", trainer["devices"])

    # Load from pretrained if specified
    if cfg.pretrained and not cfg.resume:
        print(f"Loading model from {cfg.pretrained}")
        model_module = type(model_module).load_from_checkpoint(
            cfg.pretrained, map_location="cpu", strict=cfg.strict_loading, **(model_module.hparams)
        )

    # Create checkpoint callback
    callbacks = []
    dirpath = cfg.output
    if not cfg.disable_checkpoint:
        mc = ModelCheckpoint(
            monitor="train/affinity_loss",  # Changed to train/affinity_loss for step-based monitoring
            save_top_k=cfg.save_top_k,
            save_last=True,
            mode="min",  # Minimize loss
            every_n_train_steps=cfg.save_every_n_steps,  # Save every n steps instead of epochs
            filename="train-loss-{epoch}-{step}-{train/affinity_loss:.4f}",
            auto_insert_metric_name=False,
        )
        # Add a second ModelCheckpoint
        mc_val = ModelCheckpoint(
            save_top_k=-1, 
            save_last=False,
            every_n_epochs=1, 
            filename="val-{epoch}-{step}", 
            auto_insert_metric_name=False,
        )
        callbacks = [mc, mc_val]

    # Add EarlyStopping callback if enabled
    if cfg.early_stopping and cfg.early_stopping.get("enabled", False):
        es_callback = EarlyStopping(
            monitor=cfg.early_stopping.get("monitor", "val/enzyme_all_Global_AUPRC"),
            patience=cfg.early_stopping.get("patience", 60),
            min_delta=cfg.early_stopping.get("min_delta", 0.0),
            mode=cfg.early_stopping.get("mode", "max"),
            verbose=cfg.early_stopping.get("verbose", False),
        )
        callbacks.append(es_callback)
        print(f"EarlyStopping enabled: monitor={es_callback.monitor}, patience={es_callback.patience}, mode={es_callback.mode}")

    # Create wandb logger
    loggers = []
    if wandb:
        wdb_logger = WandbLogger(
            name=wandb["name"],
            group=wandb["name"],
            save_dir=cfg.output,
            project=wandb["project"],
            entity=wandb["entity"],
            resume=wandb["resume"] if "resume" in wandb else False,
            id=wandb["id"] if "id" in wandb else None,
            log_model=False,
        )
        loggers.append(wdb_logger)

        @rank_zero_only
        def save_config_to_wandb() -> None:
            config_out = Path(wdb_logger.experiment.dir) / "run.yaml"
            with Path.open(config_out, "w") as f:
                OmegaConf.save(raw_config, f)
            wdb_logger.experiment.save(str(config_out))

        save_config_to_wandb()

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, (list, listconfig.ListConfig)) and len(devices) > 1
    ):
        strategy = DDPStrategy(find_unused_parameters=cfg.find_unused_parameters,
                               static_graph=cfg.static_graph) # Use static graph for better performance

    print(f"strategy: {strategy}")

    trainer = pl.Trainer(
        default_root_dir=str(dirpath),
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=not cfg.disable_checkpoint,
        reload_dataloaders_every_n_epochs=1,
        use_distributed_sampler=False,
        **trainer,
    )

    if not cfg.strict_loading:
        model_module.strict_loading = False

    if cfg.validation_only:
        trainer.validate(
            model_module,
            datamodule=data_module,
            ckpt_path=cfg.resume,
        )
    else:
        trainer.fit(
            model_module,
            datamodule=data_module,
            ckpt_path=cfg.resume,
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_file> [override1=value1] [override2=value2] ...")
        print("Example: python train.py configs/affinity_training.yaml data.batch_size=2")
        sys.exit(1)
    
    config_file = sys.argv[1]
    overrides = sys.argv[2:]
    train_affinity(config_file, overrides) 