
import torch
import torch.nn as nn

from evaluation.eval_utils import masked_bce_with_logits_loss, masked_mae_loss, masked_mse_loss, masked_rmse_loss


class TwoLayerMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        target: str = "accident_score",
        regression_loss: str = "mae",
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.target = target
        self.is_classification = "label" in target

        if self.is_classification:
            self.loss_fn = masked_bce_with_logits_loss
        else:
            if regression_loss == "mae":
                self.loss_fn = masked_mae_loss
            elif regression_loss == "mse":
                self.loss_fn = masked_mse_loss
            elif regression_loss == "rmse":
                self.loss_fn = masked_rmse_loss
            else:
                raise ValueError(f"Unsupported regression loss: {regression_loss}")

    def forward(self, x):
        x = self.net(x)
        if not self.is_classification:
            x = torch.sigmoid(x)
        return x

    def predict(self, x):
        out = self.forward(x)
        if self.is_classification:
            return torch.sigmoid(out)
        return out

    def get_loss(self, pred, target):
        return self.loss_fn(pred, target)

