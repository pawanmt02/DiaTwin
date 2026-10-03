"""
DiaTwin Predictive Engine

GRU-Attention model with static contextual gating for multi-horizon
quantile glucose trajectory forecasting.

Architecture:
    Static EHR → Dense Projection → Contextual Gate
                                          ↓
    Dynamic CGM → [Concat with Static] → Bi-GRU → Multi-Head Self-Attention
                                                         ↓
                                           Quantile Regression Heads (10th, 50th, 90th)
                                                         ↓
                                              [Horizon × 3] output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class StaticContextualGate(nn.Module):
    """Projects static EHR features into a dense embedding and gates dynamic inputs.

    The gate learns which aspects of the patient's static phenotype should
    modulate the temporal dynamics (e.g., insulin resistance level amplifies
    meal response patterns).
    """

    def __init__(self, static_dim: int, embedding_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(static_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_static: [batch, static_dim]
        Returns:
            embedding: [batch, embedding_dim]
        """
        return self.projection(x_static)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention over temporal GRU outputs.

    Allows the model to attend to different parts of the glucose history
    (e.g., recent meal response, overnight pattern) when making forecasts.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            attended: [batch, seq_len, hidden_dim]
        """
        B, T, D = x.shape
        residual = x

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, T, 3*D]
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, heads, T, T]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # [B, heads, T, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(B, T, D)  # [B, T, D]

        # Output projection + residual + layer norm
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class QuantileRegressionHead(nn.Module):
    """Produces multi-quantile forecasts for each horizon step.

    Outputs predictions for the 10th, 50th, and 90th percentiles,
    enabling uncertainty estimation (confidence cones).
    """

    def __init__(self, input_dim: int, horizon: int, quantiles: List[float]):
        super().__init__()
        self.horizon = horizon
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, horizon * self.num_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] context vector
        Returns:
            predictions: [batch, horizon, num_quantiles]
        """
        out = self.fc(x)  # [batch, horizon * num_quantiles]
        return out.reshape(-1, self.horizon, self.num_quantiles)


class DiaTwinForecaster(nn.Module):
    """Multi-horizon glucose trajectory forecaster with uncertainty estimation.

    Combines static patient phenotype with dynamic CGM time-series through
    contextual gating, processes with bidirectional GRU + multi-head attention,
    and outputs quantile predictions over the forecast horizon.
    """

    def __init__(
        self,
        static_dim: int = 23,
        static_embedding_dim: int = 32,
        dynamic_dim: int = 9,
        hidden_dim: int = 128,
        num_gru_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.2,
        lookback: int = 24,
        horizon: int = 24,
        quantiles: Optional[List[float]] = None,
    ):
        super().__init__()

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.hidden_dim = hidden_dim
        self.lookback = lookback
        self.horizon = horizon
        self.quantiles = quantiles

        # Static contextual gate
        self.static_gate = StaticContextualGate(static_dim, static_embedding_dim)

        # Input projection: dynamic features + static embedding concatenated
        self.input_proj = nn.Linear(
            dynamic_dim + static_embedding_dim, hidden_dim
        )

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0,
            bidirectional=True,
        )

        # Projection after BiGRU (2*hidden → hidden)
        self.bigru_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # Context aggregation (attention pooling over time)
        self.context_attn = nn.Linear(hidden_dim, 1)

        # Quantile regression head
        self.quantile_head = QuantileRegressionHead(
            input_dim=hidden_dim,
            horizon=horizon,
            quantiles=quantiles,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_static: torch.Tensor,
        x_dynamic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_static: [batch, static_dim] — patient phenotype features
            x_dynamic: [batch, lookback, dynamic_dim] — CGM time series window

        Returns:
            predictions: [batch, horizon, num_quantiles] — quantile forecasts
        """
        B, L, D = x_dynamic.shape

        # 1. Static contextual embedding
        static_emb = self.static_gate(x_static)  # [B, static_embedding_dim]

        # 2. Broadcast static embedding across time steps and concatenate
        static_expanded = static_emb.unsqueeze(1).expand(-1, L, -1)  # [B, L, emb_dim]
        fused_input = torch.cat([x_dynamic, static_expanded], dim=-1)  # [B, L, D+emb_dim]

        # 3. Project to hidden dimension
        x = self.input_proj(fused_input)  # [B, L, hidden_dim]
        x = F.relu(x)
        x = self.dropout(x)

        # 4. Bidirectional GRU
        gru_out, _ = self.gru(x)  # [B, L, 2*hidden_dim]
        gru_out = self.bigru_proj(gru_out)  # [B, L, hidden_dim]

        # 5. Multi-head self-attention
        attn_out = self.attention(gru_out)  # [B, L, hidden_dim]

        # 6. Attention-weighted temporal aggregation
        attn_weights = torch.softmax(self.context_attn(attn_out), dim=1)  # [B, L, 1]
        context = (attn_out * attn_weights).sum(dim=1)  # [B, hidden_dim]

        # 7. Quantile predictions
        predictions = self.quantile_head(context)  # [B, horizon, num_quantiles]

        return predictions


class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression.

    For each quantile q:
        L_q(y, ŷ) = max(q*(y - ŷ), (1-q)*(ŷ - y))

    This asymmetric loss penalizes under-prediction more for high quantiles
    and over-prediction more for low quantiles.
    """

    def __init__(self, quantiles: Optional[List[float]] = None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch, horizon, num_quantiles]
            targets: [batch, horizon] — raw glucose values

        Returns:
            loss: scalar
        """
        targets = targets.unsqueeze(-1)  # [batch, horizon, 1]

        losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = predictions[:, :, i: i + 1]  # [batch, horizon, 1]
            errors = targets - pred_q
            loss_q = torch.max(q * errors, (q - 1) * errors)
            losses.append(loss_q.mean())

        return sum(losses) / len(losses)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    static_dim: int = 23,
    dynamic_dim: int = 9,
    device: str = "cpu",
) -> DiaTwinForecaster:
    """Factory function to build the DiaTwin forecasting model."""
    from backend.app.core.config import config as cfg

    model = DiaTwinForecaster(
        static_dim=static_dim,
        static_embedding_dim=cfg.model.static_embedding_dim,
        dynamic_dim=dynamic_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_gru_layers=cfg.model.num_gru_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        dropout=cfg.model.dropout,
        lookback=cfg.model.lookback_window,
        horizon=cfg.model.forecast_horizon,
        quantiles=cfg.model.quantiles,
    ).to(device)

    return model
