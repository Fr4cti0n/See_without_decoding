import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedEnhancedLoss(nn.Module):
    def __init__(self, offset_weight: float = 5.0, iou_weight: float = 1.0, confidence_weight: float = 0.1):
        super().__init__()
        self.offset_weight = offset_weight
        self.iou_weight = iou_weight
        self.confidence_weight = confidence_weight

    @staticmethod
    def _compute_target_offsets(prev_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        return target_boxes - prev_boxes

    @staticmethod
    def _compute_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor):
        px1 = pred_boxes[:,0]-pred_boxes[:,2]/2; py1 = pred_boxes[:,1]-pred_boxes[:,3]/2
        px2 = pred_boxes[:,0]+pred_boxes[:,2]/2; py2 = pred_boxes[:,1]+pred_boxes[:,3]/2
        tx1 = target_boxes[:,0]-target_boxes[:,2]/2; ty1 = target_boxes[:,1]-target_boxes[:,3]/2
        tx2 = target_boxes[:,0]+target_boxes[:,2]/2; ty2 = target_boxes[:,1]+target_boxes[:,3]/2
        ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
        inter = torch.clamp(ix2-ix1, min=0)*torch.clamp(iy2-iy1, min=0)
        p_area = (px2-px1)*(py2-py1); t_area = (tx2-tx1)*(ty2-ty1)
        union = p_area + t_area - inter + 1e-8
        iou = inter / union
        return 1 - iou.mean(), iou.mean(), iou

    def forward(self, predictions, targets):
        pred_offsets = predictions['predicted_offsets']
        pred_confidences = predictions['predicted_confidences']
        updated_boxes = predictions['updated_boxes']
        prev_boxes = targets['prev_boxes']
        target_boxes = targets['target_boxes']
        target_offsets = self._compute_target_offsets(prev_boxes, target_boxes)
        offset_diff = torch.abs(pred_offsets - target_offsets)
        weighted_offset_loss = (offset_diff * pred_confidences).mean()
        iou_loss, mean_iou, individual_ious = self._compute_iou(updated_boxes, target_boxes)
        confidence_reg = F.mse_loss(pred_confidences.mean(1), torch.clamp(mean_iou.detach().expand(len(pred_confidences)), 0.2, 0.9))
        total = self.offset_weight * weighted_offset_loss + self.iou_weight * iou_loss + self.confidence_weight * confidence_reg
        return total, { 'total_loss': float(total.item()), 'offset_loss': float(weighted_offset_loss.item()), 'iou_loss': float(iou_loss.item()), 'confidence_reg': float(confidence_reg.item()), 'mean_iou': float(mean_iou.item()), 'individual_ious': individual_ious }
