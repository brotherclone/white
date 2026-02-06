#!/usr/bin/env python3
"""
Embedding Loader for Phase 4 Training

Provides utilities for:
1. Loading pre-computed embeddings from parquet files (for training)
2. Computing embeddings on-the-fly using DeBERTa (for inference/validation)

Usage:
    # For training - load pre-computed embeddings
    loader = PrecomputedEmbeddingLoader("/path/to/training_segments_media.parquet")
    embedding = loader.get_embedding(segment_id)

    # For inference - compute embeddings from text
    encoder = DeBERTaEmbeddingEncoder()
    embedding = encoder.encode("concept text here")
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Union


class PrecomputedEmbeddingLoader:
    """
    Loads pre-computed embeddings from parquet file.

    Use this for training when embeddings are already extracted.
    """

    def __init__(
        self,
        parquet_path: Union[str, Path],
        id_column: str = "segment_id",
        embedding_column: str = "embedding",
    ):
        """
        Initialize the embedding loader.

        Args:
            parquet_path: Path to parquet file with embeddings
            id_column: Column name for segment IDs (for lookup)
            embedding_column: Column name containing embeddings
        """
        self.parquet_path = Path(parquet_path)
        self.id_column = id_column
        self.embedding_column = embedding_column

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        print(f"Loading embeddings from {self.parquet_path}...")
        self.df = pd.read_parquet(self.parquet_path)

        if self.embedding_column not in self.df.columns:
            raise ValueError(
                f"Embedding column '{self.embedding_column}' not found. "
                f"Available columns: {list(self.df.columns)}"
            )

        # Create index for fast lookup by ID
        if self.id_column in self.df.columns:
            self._id_to_idx = {
                str(row[self.id_column]): idx for idx, row in self.df.iterrows()
            }
            print(f"Created ID index with {len(self._id_to_idx)} entries")
        else:
            self._id_to_idx = None
            print(f"No '{self.id_column}' column - using positional indexing only")

        # Verify embedding dimensions
        sample_embedding = self.df[self.embedding_column].iloc[0]
        self.embedding_dim = len(np.array(sample_embedding))
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Total segments: {len(self.df)}")

    def get_embedding_by_idx(self, idx: int) -> np.ndarray:
        """Get embedding by positional index."""
        if idx < 0 or idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range [0, {len(self.df)})")

        embedding = self.df[self.embedding_column].iloc[idx]
        return np.array(embedding, dtype=np.float32)

    def get_embedding_by_id(self, segment_id: str) -> np.ndarray:
        """Get embedding by segment ID."""
        if self._id_to_idx is None:
            raise ValueError("ID-based lookup not available - no ID column in parquet")

        segment_id = str(segment_id)
        if segment_id not in self._id_to_idx:
            raise KeyError(f"Segment ID '{segment_id}' not found in embeddings")

        idx = self._id_to_idx[segment_id]
        return self.get_embedding_by_idx(idx)

    def get_embedding(
        self, idx: Optional[int] = None, segment_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Get embedding by index or ID.

        Args:
            idx: Positional index in the dataframe
            segment_id: Segment ID for lookup

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if idx is not None:
            return self.get_embedding_by_idx(idx)
        elif segment_id is not None:
            return self.get_embedding_by_id(segment_id)
        else:
            raise ValueError("Must provide either idx or segment_id")

    def __len__(self) -> int:
        return len(self.df)


class DeBERTaEmbeddingEncoder:
    """
    Computes embeddings on-the-fly using DeBERTa-v3-base.

    Use this for inference/validation when processing new concepts
    that don't have pre-computed embeddings.
    """

    MODEL_NAME = "microsoft/deberta-v3-base"

    def __init__(
        self,
        device: str = "cpu",
        max_length: int = 512,
    ):
        """
        Initialize the DeBERTa encoder.

        Args:
            device: Device to run on ('cpu', 'cuda', 'mps')
            max_length: Maximum sequence length
        """
        self.device = device
        self.max_length = max_length

        # Lazy load to avoid import overhead when not needed
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """Load the model and tokenizer (lazy initialization)."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModel

        print("Loading DeBERTa tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        print("Loading DeBERTa model...")
        # Use safetensors if available for security
        try:
            self._model = AutoModel.from_pretrained(
                self.MODEL_NAME, use_safetensors=True
            )
        except Exception:
            self._model = AutoModel.from_pretrained(self.MODEL_NAME)

        self._model = self._model.to(self.device)
        self._model.eval()

        print(f"DeBERTa loaded on {self.device}")
        print(f"Hidden size: {self._model.config.hidden_size}")

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (768 for DeBERTa-v3-base)."""
        return 768

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text to embedding.

        Args:
            text: Text to encode

        Returns:
            numpy array of shape (768,)
        """
        self._load_model()

        if not text or pd.isna(text):
            text = ""

        with torch.no_grad():
            # Tokenize
            encoded = self._tokenizer(
                str(text),
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Move to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Forward pass
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)

            # CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding[0].astype(np.float32)

    def encode_batch(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing

        Returns:
            numpy array of shape (len(texts), 768)
        """
        self._load_model()

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = [
                str(t) if t and not pd.isna(t) else ""
                for t in texts[i : i + batch_size]
            ]

            with torch.no_grad():
                encoded = self._tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                outputs = self._model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)

                # Free GPU memory
                del input_ids, attention_mask, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return np.array(embeddings, dtype=np.float32)


def find_embedding_file(data_dir: Union[str, Path]) -> Optional[Path]:
    """
    Find the best available embedding file in a directory.

    Checks for files in order of preference:
    1. training_data_with_embeddings.parquet (DeBERTa text embeddings, smaller)
    2. training_segments_media.parquet (full media: audio waveforms + MIDI binary)

    Args:
        data_dir: Directory to search

    Returns:
        Path to embedding file, or None if not found
    """
    data_dir = Path(data_dir)

    candidates = [
        "training_data_with_embeddings.parquet",
        "training_segments_media.parquet",
    ]

    for filename in candidates:
        path = data_dir / filename
        if path.exists():
            size_gb = path.stat().st_size / (1024**3)
            print(f"Found embedding file: {path} ({size_gb:.2f} GB)")
            return path

    return None
