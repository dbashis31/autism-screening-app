"""Shared constants for all governance agents."""

APPROVED_MODEL_REGISTRY: set[str] = {
    "model-v2.1-signed",
    "model-v2.0-signed",
    "asd-cnnrnn-v1",          # CNN-BiLSTM multimodal model (MLHC 2026)
}

APPROVED_CAREGIVER_VOCAB: list[str] = [
    "Your child's developmental screen is complete. A clinician will review and follow up with you.",
    "Additional information is needed before results can be shared. A clinician will contact you.",
    "Your child's screening session has been logged. No further action is needed at this time.",
]
