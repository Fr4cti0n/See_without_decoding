from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path
from .logging import ProgressLogger
from ..metrics import iou_diagonal, simple_map

@dataclass
class EpochResult:
    loss: float
    iou: float
    map: float

class SlimTrainer:
    def __init__(self, model, criterion, optimizer, device='cpu', scheduler=None, grad_clip: float = 1.0, output_dir: str = 'outputs', ckpt_name: str = 'best_slim.pt', maximize_metric: str = 'map'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_name = ckpt_name
        self.logger = ProgressLogger()
        self.maximize_metric = maximize_metric
        self.best_metric = -1.0
        self.best_state: Dict[str, Any] | None = None

    def _step_batch(self, batch: Dict[str, Any], train: bool = True):
        mv_maps = batch['mv_map'].to(self.device)
        losses = []
        ious = []
        maps = []
        if train:
            self.optimizer.zero_grad()
        for i, prev_boxes in enumerate(batch['prev_boxes']):
            prev_boxes = prev_boxes.to(self.device)
            tgt_boxes = batch['target_boxes'][i].to(self.device)
            if len(prev_boxes) == 0:
                continue
            preds = self.model(mv_maps[i:i+1], prev_boxes)
            loss, comps = self.criterion(preds, {'prev_boxes': prev_boxes, 'target_boxes': tgt_boxes})
            losses.append(loss)
            ious.append(comps['mean_iou'])
            maps.append(simple_map(preds['updated_boxes'].detach(), tgt_boxes.detach()))
            if train:
                loss.backward()
        if train and losses:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        if not losses:
            return 0.0, 0.0, 0.0
        return (float(torch.stack(losses).mean().item()),
                float(sum(ious)/len(ious)),
                float(sum(maps)/len(maps)))

    def run_epoch(self, loader: DataLoader, train: bool = True) -> EpochResult:
        self.model.train(mode=train)
        total_loss = []
        total_iou = []
        total_map = []
        for batch in loader:
            if batch is None:
                continue
            loss, iou, m = self._step_batch(batch, train=train)
            if loss == 0 and iou == 0 and m == 0:
                continue
            total_loss.append(loss); total_iou.append(iou); total_map.append(m)
        if self.scheduler and not train:
            # step on validation loss
            self.scheduler.step(sum(total_loss)/len(total_loss))
        if not total_loss:
            return EpochResult(0.0, 0.0, 0.0)
        return EpochResult(sum(total_loss)/len(total_loss), sum(total_iou)/len(total_iou), sum(total_map)/len(total_map))

    def maybe_checkpoint(self, epoch: int, result: EpochResult):
        metric_value = getattr(result, self.maximize_metric)
        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_state = {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'epoch': epoch,
                'best_metric': self.best_metric,
                'metric_name': self.maximize_metric,
            }
            torch.save(self.best_state, self.output_dir / self.ckpt_name)
            print(f"  💾 New best {self.maximize_metric}={metric_value:.4f} saved at epoch {epoch}")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        history = []
        for e in range(1, epochs+1):
            tr = self.run_epoch(train_loader, train=True)
            vr = self.run_epoch(val_loader, train=False)
            self.logger.epoch(e, epochs, {'loss': tr.loss, 'iou': tr.iou, 'map': tr.map}, {'loss': vr.loss, 'iou': vr.iou, 'map': vr.map})
            self.maybe_checkpoint(e, vr)
            history.append({'epoch': e, 'train': tr.__dict__, 'val': vr.__dict__})
        return history
