"""
CNN-BiLSTM multimodal ASD screening model for MLHC 2026.

Architecture:
  Stage 1 — CNN Encoders (per modality, applied per-visit after B*T reshape)
  Stage 2 — BiLSTM Temporal Encoding (4 separate towers, one per modality)
  Stage 3 — Late Fusion (cat → FusionMLP)
  Stage 4 — Per-modality heads + Global head (all sigmoid outputs)

Tensor shape contracts:
  B = batch size, T = number of temporal visits (default 5)
  images:        (B, T, 3, 64, 64)
  audio:         (B, T, 1, 40, 128)
  questionnaire: (B, T, 10)
  text:          (B, T, 50)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ── Stage 1: CNN Encoders ─────────────────────────────────────────────────────

class VisualCNNEncoder(nn.Module):
    """
    ResNet18 backbone → 256-d frame embedding.
    Input:  (B*T, 3, 64, 64)
    Output: (B*T, 256)
    """
    def __init__(self, out_dim: int = 256, dropout: float = 0.3,
                 freeze_early: bool = True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final FC layer; keep everything up to avgpool
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Freeze early layers (conv1 through layer2) for efficiency
        if freeze_early:
            for layer in [backbone.conv1, backbone.bn1,
                          backbone.layer1, backbone.layer2]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, 3, 64, 64)
        x = self.features(x)          # (B*T, 512, 2, 2)
        x = self.pool(x)              # (B*T, 512, 1, 1)
        x = x.flatten(1)             # (B*T, 512)
        return self.proj(x)           # (B*T, 256)


class AudioCNNEncoder(nn.Module):
    """
    3-layer 2D CNN on log-mel spectrograms → 256-d frame embedding.
    Input:  (B*T, 1, 40, 128)
    Output: (B*T, 256)
    """
    def __init__(self, out_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            # Block 1: 1→32
            nn.Conv2d(1,  32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # (B*T, 32, 20, 64)
            # Block 2: 32→64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # (B*T, 64, 10, 32)
            # Block 3: 64→128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # (B*T, 128, 5, 16)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, 1, 40, 128)
        x = self.cnn(x)    # (B*T, 128, 5, 16)
        x = self.pool(x)   # (B*T, 128, 1, 1)
        x = x.flatten(1)   # (B*T, 128)
        return self.proj(x) # (B*T, 256)


class QuestionnaireMLP(nn.Module):
    """
    3-layer MLP on M-CHAT style features → 256-d embedding.
    Input:  (B*T, 10)
    Output: (B*T, 256)
    """
    def __init__(self, in_dim: int = 10, out_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 128),   nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, out_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # (B*T, 256)


class TextEncoder(nn.Module):
    """
    2-layer MLP projecting mean-pool text embeddings → 256-d.
    Input:  (B*T, 50)
    Output: (B*T, 256)
    """
    def __init__(self, in_dim: int = 50, out_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)  # (B*T, 256)


# ── Stage 2: BiLSTM Temporal Encoding ────────────────────────────────────────

class ModalityBiLSTM(nn.Module):
    """
    2-layer BiLSTM for temporal encoding of one modality.
    Input:  (B, T, 256)
    Output: (B, 128)  — last-timestep hidden state projected down
    """
    def __init__(self, input_size: int = 256, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 128 * 2 directions = 256 → project down to 128
        self.proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 256)
        out, _ = self.lstm(x)       # (B, T, 256) [128*2 bidir]
        last = out[:, -1, :]        # (B, 256) — take last timestep
        return self.proj(last)      # (B, 128)


# ── Stage 4: Output Heads ─────────────────────────────────────────────────────

class ModalityHead(nn.Module):
    """
    Per-modality sigmoid head: concat(fused_repr, mod_enc) → scalar.
    Input:  fused(B,256) + mod_enc(B,128) → cat → (B,384)
    Output: (B, 1) in [0,1]
    """
    def __init__(self, in_dim: int = 384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, fused: torch.Tensor,
                mod_enc: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fused, mod_enc], dim=1)  # (B, 384)
        return self.head(x)                       # (B, 1)


class GlobalHead(nn.Module):
    """
    Overall ASD risk head.
    Input:  (B, 256)
    Output: (B, 1) in [0,1]
    """
    def __init__(self, in_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)  # (B, 1)


# ── Full Model ────────────────────────────────────────────────────────────────

class ASDScreeningModel(nn.Module):
    """
    Full CNN-BiLSTM multimodal ASD screening model.

    Forward inputs (all tensors, same device):
      images:         (B, T, 3, 64, 64)
      audio:          (B, T, 1, 40, 128)
      questionnaire:  (B, T, 10)
      text:           (B, T, 50)

    Forward returns:
      {
        "global":        (B, 1),
        "audio":         (B, 1),
        "video":         (B, 1),
        "questionnaire": (B, 1),
        "text":          (B, 1),
      }
    """
    MODALITIES: list = ["audio", "video", "questionnaire", "text"]

    def __init__(
        self,
        out_dim:       int   = 256,
        lstm_hidden:   int   = 128,
        lstm_layers:   int   = 2,
        dropout:       float = 0.3,
        freeze_resnet: bool  = True,
    ):
        super().__init__()
        self.hparams = dict(
            out_dim=out_dim, lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers, dropout=dropout,
        )

        # Stage 1 — CNN encoders
        self.visual_encoder        = VisualCNNEncoder(out_dim, dropout, freeze_resnet)
        self.audio_encoder         = AudioCNNEncoder(out_dim, dropout)
        self.questionnaire_encoder = QuestionnaireMLP(10, out_dim, dropout)
        self.text_encoder          = TextEncoder(50, out_dim, dropout)

        # Stage 2 — BiLSTM temporal towers (one per modality)
        self.audio_rnn         = ModalityBiLSTM(out_dim, lstm_hidden, lstm_layers, dropout)
        self.visual_rnn        = ModalityBiLSTM(out_dim, lstm_hidden, lstm_layers, dropout)
        self.questionnaire_rnn = ModalityBiLSTM(out_dim, lstm_hidden, lstm_layers, dropout)
        self.text_rnn          = ModalityBiLSTM(out_dim, lstm_hidden, lstm_layers, dropout)

        # Stage 3 — Late fusion: 4 × 128 = 512 → 256
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # Stage 4 — Heads
        fusion_plus_enc = out_dim + lstm_hidden   # 256 + 128 = 384
        self.head_audio         = ModalityHead(fusion_plus_enc)
        self.head_video         = ModalityHead(fusion_plus_enc)
        self.head_questionnaire = ModalityHead(fusion_plus_enc)
        self.head_text          = ModalityHead(fusion_plus_enc)
        self.head_global        = GlobalHead(out_dim)

    def _encode_cnn(self, encoder, x: torch.Tensor, T: int) -> torch.Tensor:
        """Apply a CNN encoder frame-by-frame: (B,T,C,...) → (B,T,256)."""
        B = x.shape[0]
        # Flatten visits into batch dim
        flat_shape = (B * T,) + x.shape[2:]
        flat = x.reshape(flat_shape)
        enc = encoder(flat)           # (B*T, 256)
        return enc.view(B, T, -1)     # (B, T, 256)

    def forward(
        self,
        images:        torch.Tensor,
        audio:         torch.Tensor,
        questionnaire: torch.Tensor,
        text:          torch.Tensor,
    ) -> dict:
        B, T = images.shape[:2]

        # Stage 1 — per-visit CNN encoding
        vid_seq  = self._encode_cnn(self.visual_encoder,        images,        T)  # (B,T,256)
        aud_seq  = self._encode_cnn(self.audio_encoder,         audio,         T)  # (B,T,256)
        qst_seq  = self._encode_cnn(self.questionnaire_encoder, questionnaire, T)  # (B,T,256)
        txt_seq  = self._encode_cnn(self.text_encoder,          text,          T)  # (B,T,256)

        # Stage 2 — BiLSTM temporal encoding
        aud_enc = self.audio_rnn(aud_seq)          # (B, 128)
        vid_enc = self.visual_rnn(vid_seq)          # (B, 128)
        qst_enc = self.questionnaire_rnn(qst_seq)  # (B, 128)
        txt_enc = self.text_rnn(txt_seq)            # (B, 128)

        # Stage 3 — Late fusion
        fused = torch.cat([aud_enc, vid_enc, qst_enc, txt_enc], dim=1)  # (B, 512)
        shared = self.fusion(fused)                                       # (B, 256)

        # Stage 4 — Heads
        return {
            "audio":         self.head_audio(shared, aud_enc),         # (B, 1)
            "video":         self.head_video(shared, vid_enc),         # (B, 1)
            "questionnaire": self.head_questionnaire(shared, qst_enc), # (B, 1)
            "text":          self.head_text(shared, txt_enc),          # (B, 1)
            "global":        self.head_global(shared),                 # (B, 1)
        }

    def enable_dropout(self) -> None:
        """Set all Dropout layers to train mode for MC Dropout inference."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
