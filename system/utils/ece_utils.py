import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError

class ECEEvaluator:
    def __init__(self, num_classes, device='cpu', n_bins=15, norm='l1'):
        self.ece_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins, norm=norm).to(device)
        self.device = device

    def compute_ece(self, logits, labels):
        """
        Compute Expected Calibration Error (ECE).
        logits: tensor of raw model outputs, shape [N, num_classes]
        labels: tensor of true labels, shape [N]
        """
        probs = F.softmax(logits, dim=1)
        ece = self.ece_metric(probs, labels).item()
        return ece
