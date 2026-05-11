import torch
import torch.nn as nn


class WeightedMotionReconstructionLoss(nn.Module):
    def __init__(
        self,
        velocity_weight: float = 0.7,
        acceleration_weight: float = 0.2,
        root_velocity_weight: float = 0.2,
        vq_loss_weight: float = 0.25,
    ):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.root_velocity_weight = root_velocity_weight
        self.vq_loss_weight = vq_loss_weight
        weights = torch.ones(182, dtype=torch.float32)
        weights[0:3] = 0.2
        weights[3:66] = 1.2
        weights[66:111] = 4.0
        weights[111:156] = 4.0
        weights[156:159] = 0.2
        weights[159:169] = 0.0
        weights[169:179] = 0.1
        weights[179:182] = 0.0
        self.register_buffer("weights", weights.view(1, 1, -1))

    def _weighted_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target).pow(2) * self.weights).mean()

    @staticmethod
    def _mse(pred: torch.Tensor, target: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return (pred[..., start:end] - target[..., start:end]).pow(2).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        vq_loss: torch.Tensor | None = None,
        perplexity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}")

        zero = pred.new_tensor(0.0)
        vq = vq_loss if vq_loss is not None else zero
        recon_loss = self._weighted_mse(pred, target)
        velocity_loss = self._weighted_mse(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
        if pred.shape[1] > 2:
            pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
            target_acc = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
            acceleration_loss = self._weighted_mse(pred_acc, target_acc)
        else:
            acceleration_loss = zero
        root_velocity_loss = (pred[:, 1:, 0:3] - pred[:, :-1, 0:3] - (target[:, 1:, 0:3] - target[:, :-1, 0:3])).pow(2).mean()
        loss = (
            recon_loss
            + self.velocity_weight * velocity_loss
            + self.acceleration_weight * acceleration_loss
            + self.vq_loss_weight * vq
            + self.root_velocity_weight * root_velocity_loss
        )

        metrics = {
            "loss": loss,
            "recon_loss": recon_loss,
            "velocity_loss": velocity_loss,
            "acceleration_loss": acceleration_loss,
            "vq_loss": vq,
            "perplexity": perplexity if perplexity is not None else zero,
            "root_velocity_loss": root_velocity_loss,
            "root_mse": self._mse(pred, target, 0, 3),
            "body_mse": self._mse(pred, target, 3, 66),
            "left_hand_mse": self._mse(pred, target, 66, 111),
            "right_hand_mse": self._mse(pred, target, 111, 156),
            "jaw_mse": self._mse(pred, target, 156, 159),
            "betas_mse": self._mse(pred, target, 159, 169),
            "expression_mse": self._mse(pred, target, 169, 179),
            "camera_mse": self._mse(pred, target, 179, 182),
        }
        return metrics
