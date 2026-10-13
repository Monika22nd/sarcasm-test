"""Lightweight NumPy inference for the exported Bidirectional LSTM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


class SarcasmPredictor:
    """Load exported artifacts and reproduce the Keras inference path."""

    def __init__(self, artifact_dir: str | Path) -> None:
        artifact_dir = Path(artifact_dir)

        with (artifact_dir / "config.json").open(encoding="utf-8") as file:
            self.config = json.load(file)

        with (artifact_dir / "tokenizer_min.json").open(encoding="utf-8") as file:
            tokenizer = json.load(file)

        self.word_index = tokenizer["word_index"]
        self.num_words = int(tokenizer["num_words"])
        self.oov_index = int(tokenizer["word_index"][tokenizer["oov_token"]])
        self.lower = bool(tokenizer["lower"])
        self.filters = tokenizer["filters"]
        self.split = tokenizer["split"]
        self.max_len = int(self.config["max_len"])

        weights = np.load(artifact_dir / "weights.npz")
        self.embedding = weights["embedding"].astype(np.float32, copy=False)
        self.forward_kernel = weights["forward_kernel"].astype(np.float32, copy=False)
        self.forward_recurrent = weights["forward_recurrent"].astype(np.float32, copy=False)
        self.forward_bias = weights["forward_bias"].astype(np.float32, copy=False)
        self.backward_kernel = weights["backward_kernel"].astype(np.float32, copy=False)
        self.backward_recurrent = weights["backward_recurrent"].astype(np.float32, copy=False)
        self.backward_bias = weights["backward_bias"].astype(np.float32, copy=False)
        self.dense_kernel = weights["dense_kernel"].astype(np.float32, copy=False)
        self.dense_bias = weights["dense_bias"].astype(np.float32, copy=False)
        self.output_kernel = weights["output_kernel"].astype(np.float32, copy=False)
        self.output_bias = weights["output_bias"].astype(np.float32, copy=False)

        self._translation = str.maketrans({character: self.split for character in self.filters})

    def tokenize(self, text: str) -> np.ndarray:
        """Apply the same normalization, truncation and post-padding as Keras."""
        if self.lower:
            text = text.lower()
        words = text.translate(self._translation).split(self.split)

        sequence: list[int] = []
        for word in words:
            if not word:
                continue
            index = int(self.word_index.get(word, self.oov_index))
            if index < self.num_words:
                sequence.append(index)
            else:
                sequence.append(self.oov_index)

        sequence = sequence[-self.max_len :]
        padded = np.zeros(self.max_len, dtype=np.int32)
        padded[: len(sequence)] = sequence
        return padded

    @staticmethod
    def _lstm(
        embedded: np.ndarray,
        token_ids: np.ndarray,
        kernel: np.ndarray,
        recurrent_kernel: np.ndarray,
        bias: np.ndarray,
        reverse: bool = False,
    ) -> np.ndarray:
        units = recurrent_kernel.shape[0]
        hidden = np.zeros(units, dtype=np.float32)
        cell = np.zeros(units, dtype=np.float32)
        positions = range(len(token_ids) - 1, -1, -1) if reverse else range(len(token_ids))

        for position in positions:
            if token_ids[position] == 0:  # Embedding(mask_zero=True)
                continue
            projected = embedded[position] @ kernel + hidden @ recurrent_kernel + bias
            input_gate, forget_gate, candidate, output_gate = np.split(projected, 4)
            input_gate = _sigmoid(input_gate)
            forget_gate = _sigmoid(forget_gate)
            output_gate = _sigmoid(output_gate)
            candidate = np.tanh(candidate)
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * np.tanh(cell)

        return hidden

    def predict_probability(self, text: str) -> float:
        token_ids = self.tokenize(text)
        embedded = self.embedding[token_ids]

        forward = self._lstm(
            embedded,
            token_ids,
            self.forward_kernel,
            self.forward_recurrent,
            self.forward_bias,
        )
        backward = self._lstm(
            embedded,
            token_ids,
            self.backward_kernel,
            self.backward_recurrent,
            self.backward_bias,
            reverse=True,
        )

        features = np.concatenate((forward, backward))
        dense = np.maximum(features @ self.dense_kernel + self.dense_bias, 0.0)
        logit = dense @ self.output_kernel + self.output_bias
        return float(_sigmoid(logit).item())

    def predict(self, text: str, threshold: float = 0.5) -> dict[str, float | str]:
        probability = self.predict_probability(text)
        return {
            "label": "Sarcastic" if probability >= threshold else "Not sarcastic",
            "probability": probability,
            "threshold": threshold,
        }
