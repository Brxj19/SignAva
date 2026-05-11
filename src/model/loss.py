import torch
import torch.nn as nn


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


def velocity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_vel = pred[:, 1:] - pred[:, :-1]  # (batch, seq_len-1, motion_dim)
    target_vel = target[:, 1:] - target[:, :-1]
    return nn.functional.mse_loss(pred_vel, target_vel)


def acceleration_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]  # (batch, seq_len-2, motion_dim)
    target_acc = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
    return nn.functional.mse_loss(pred_acc, target_acc)


class SignMotionLoss(nn.Module):
    def __init__(
        self,
        mse_weight: float = 1.0,
        velocity_weight: float = 0.5,
        acceleration_weight: float = 0.1,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        mse = mse_loss(pred, target)
        vel = velocity_loss(pred, target)
        acc = acceleration_loss(pred, target)

        total_loss = self.mse_weight * mse + self.velocity_weight * vel + self.acceleration_weight * acc

        return {
            "loss": total_loss,
            "mse": mse,
            "velocity": vel,
            "acceleration": acc,
        }


SMPLX_SLICES = {
    "root": slice(0, 3),
    "body": slice(3, 66),
    "left_hand": slice(66, 111),
    "right_hand": slice(111, 156),
    "jaw": slice(156, 159),
    "betas": slice(159, 169),
    "expression": slice(169, 179),
    "translation": slice(179, 182),
}


class WeightedSignMotionLoss(nn.Module):
    def __init__(
        self,
        velocity_weight: float = 0.7,
        acceleration_weight: float = 0.2,
        part_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.part_weights = part_weights or {
            "root": 0.5,
            "body": 1.0,
            "left_hand": 4.0,
            "right_hand": 4.0,
            "jaw": 0.2,
            "betas": 0.0,
            "expression": 0.1,
            "translation": 0.0,
        }

    def _part_mse(self, pred: torch.Tensor, target: torch.Tensor, part: str) -> torch.Tensor:
        part_slice = SMPLX_SLICES[part]
        return nn.functional.mse_loss(pred[..., part_slice], target[..., part_slice])

    def _weighted_temporal_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = pred.new_tensor(0.0)
        total_weight = 0.0
        for part, part_slice in SMPLX_SLICES.items():
            weight = float(self.part_weights.get(part, 0.0))
            if weight <= 0.0:
                continue
            mse = nn.functional.mse_loss(pred[..., part_slice], target[..., part_slice])
            total = total + weight * mse
            total_weight += weight
        if total_weight <= 0.0:
            raise ValueError("At least one weighted loss part must have a positive weight.")
        return total / total_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        weighted_mse = self._weighted_temporal_mse(pred, target)

        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        weighted_velocity = self._weighted_temporal_mse(pred_vel, target_vel)

        pred_acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        target_acc = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
        weighted_acceleration = self._weighted_temporal_mse(pred_acc, target_acc)

        total_loss = (
            weighted_mse
            + self.velocity_weight * weighted_velocity
            + self.acceleration_weight * weighted_acceleration
        )

        return {
            "loss": total_loss,
            "weighted_mse": weighted_mse,
            "weighted_velocity": weighted_velocity,
            "weighted_acceleration": weighted_acceleration,
            "root_mse": self._part_mse(pred, target, "root"),
            "body_mse": self._part_mse(pred, target, "body"),
            "left_hand_mse": self._part_mse(pred, target, "left_hand"),
            "right_hand_mse": self._part_mse(pred, target, "right_hand"),
            "jaw_mse": self._part_mse(pred, target, "jaw"),
            "betas_mse": self._part_mse(pred, target, "betas"),
            "expression_mse": self._part_mse(pred, target, "expression"),
            "translation_mse": self._part_mse(pred, target, "translation"),
        }
