"""
Affinity Prediction Writer

This module provides a custom writer for affinity predictions that maintains a real-time updated dictionary and writes results immediately after each batch.
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor
import os
import torch.distributed as dist


class AffinityPredictionWriter(BasePredictionWriter):

    def __init__(
        self,
        output_dir: str,
        output_filename: str = "affinity_predictions",
        task: str = "binary",
    ) -> None:
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.final_output_path = self.output_dir / f"{output_filename}.json"
        self.task = task
        self.results: dict[str, Any] = {}

    def _filter_fields(self, result: dict) -> dict:
        """Filter result fields based on task."""
        base = {"status": result.get("status")}
        if self.task == "binary":
            base["binary"] = result.get("binary")
        elif self.task == "value":
            base["pred_value"] = result.get("pred_value")
            base["pred_value_raw"] = result.get("pred_value_raw")
            base["mw"] = result.get("mw")
        elif self.task == "enzyme":
            base["enzyme"] = result.get("enzyme")
        return base

    def _decode_record_id(self, batch: Dict[str, Tensor], batch_idx: int) -> str:
        record_id_tensor = batch.get("record_id")
        if isinstance(record_id_tensor, torch.Tensor):
            try:
                chars = [chr(int(x)) for x in record_id_tensor.cpu().tolist()]
                name = "".join(chars).rstrip("-")
                if name:
                    return name
            except Exception:
                pass
        rec_list = batch.get("record")
        if isinstance(rec_list, (list, tuple)) and rec_list:
            try:
                return getattr(rec_list[0], "id", f"unknown_batch_{batch_idx}")
            except Exception:
                return f"unknown_batch_{batch_idx}"
        return f"unknown_batch_{batch_idx}"

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Dict[str, Tensor],
        batch_indices: list[int],
        batch: Dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        record_id = self._decode_record_id(batch, batch_idx)
        
        if prediction.get("exception", False):
            result = {"status": "failed", "binary": None, "pred_value": None, "pred_value_raw": None, "enzyme": None, "mw": None}
        else:
            result = {
                "status": "success",
                "pred_value": float(prediction["affinity_pred_value"].item()) if "affinity_pred_value" in prediction else None,
                "pred_value_raw": float(prediction["affinity_pred_value_raw"].item()) if "affinity_pred_value_raw" in prediction else None,
                "binary": float(prediction["affinity_probability_binary"].item()) if "affinity_probability_binary" in prediction else None,
                "enzyme": float(prediction["affinity_probability_enzyme"].item()) if "affinity_probability_enzyme" in prediction else None,
                "mw": float(prediction["mw"].item()) if "mw" in prediction else None,
            }
        
        self.results[record_id] = self._filter_fields(result)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # DDP: gather all results to rank0
        if dist.is_available() and dist.is_initialized():
            all_results = [None] * dist.get_world_size()
            dist.all_gather_object(all_results, self.results)
            if dist.get_rank() == 0:
                merged = {}
                for r in all_results:
                    merged.update(r)
                self.results = merged
            else:
                return  
        
        with self.final_output_path.open("w") as f:
            json.dump(self.results, f, indent=2)
        
        total = len(self.results)
        success = sum(1 for r in self.results.values() if r.get("status") == "success")
        print(f"\n{'='*60}")
        print(f"PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {total}, Success: {success}, Failed: {total - success}")
        print(f"Results saved to: {self.final_output_path}")
        print(f"{'='*60}")