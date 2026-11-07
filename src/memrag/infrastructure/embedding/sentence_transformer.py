"""EmbeddingProvider backed by Sentence Transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from memrag.domain.schemas.embedding import EncodedText, ModelSpec, TextKind


class SentenceTransformerProvider:
    def __init__(self, spec: ModelSpec, model_path: Path, device: str = "cuda", batch_size: int = 32):
        import torch
        from sentence_transformers import SentenceTransformer

        # CPU always float32: the Nano's Cortex-A78AE has no bf16 instructions
        dtype = getattr(torch, spec.dtype) if device.startswith("cuda") else torch.float32
        self._spec = spec
        self._batch_size = batch_size
        self.model = SentenceTransformer(
            str(model_path),
            device=device,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            model_kwargs={"dtype": dtype},
        )
        self.model.max_seq_length = spec.max_seq_length

    def info(self) -> ModelSpec:
        return self._spec

    def explain(self, text: str, kind: TextKind) -> EncodedText:
        return self._spec.encode_text(text, kind)

    def embed_queries(self, texts: Sequence[str]):
        return self._embed(texts, "query")

    def embed_documents(self, texts: Sequence[str]):
        return self._embed(texts, "document")

    def _embed(self, texts: Sequence[str], kind: TextKind):
        import numpy as np

        encoded = [self._spec.encode_text(t, kind).encoded_text for t in texts]
        emb = self.model.encode(
            encoded,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        # Normalize in float32: normalizing in bf16 leaves norms off by ~0.1%, which skews cosine scores
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)
