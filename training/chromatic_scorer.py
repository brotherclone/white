"""
ChromaticScorer — fitness function for evolutionary music composition.

Wraps the ONNX-exported multimodal fusion model with lazy-loaded DeBERTa
and CLAP encoders. Designed for CPU inference with batch scoring of 50+
candidates per evolutionary stage.

Usage:
    scorer = ChromaticScorer()

    # Encode concept text once, reuse across all candidates
    concept_emb = scorer.prepare_concept("RED temporal=Past spatial=Thing ontological=Known")

    # Score a single candidate
    result = scorer.score(midi_bytes=midi_data, concept_emb=concept_emb)
    # → {"temporal": {"past": 0.8, ...}, "spatial": {...}, "ontological": {...}, "confidence": 0.89}

    # Score a batch of 50 candidates
    candidates = [{"midi_bytes": m} for m in midi_variants]
    ranked = scorer.score_batch(candidates, concept_emb=concept_emb)
    # → sorted list with scores + original candidate data
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

# Import midi_bytes_to_piano_roll without triggering training.models.__init__
# (which eagerly imports torch-dependent classifiers)
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "piano_roll_encoder",
    str(Path(__file__).parent / "models" / "piano_roll_encoder.py"),
)
_pre = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pre)
midi_bytes_to_piano_roll = _pre.midi_bytes_to_piano_roll

logger = logging.getLogger(__name__)

# Mode labels matching the training data order
TEMPORAL_MODES = ["Past", "Present", "Future"]
SPATIAL_MODES = ["Thing", "Place", "Person"]
ONTOLOGICAL_MODES = ["Imagined", "Forgotten", "Known"]


class ChromaticScorer:
    """Scores audio/MIDI candidates against chromatic concepts.

    The ONNX model (16 MB) loads immediately. DeBERTa (~400 MB) and CLAP
    (~600 MB) are lazy-loaded on first use. For MIDI-only scoring (the
    primary evolutionary use case), CLAP never loads.
    """

    def __init__(self, onnx_path: Optional[str] = None):
        """Initialize the scorer with an ONNX model.

        Args:
            onnx_path: Path to fusion_model.onnx. Defaults to
                training/data/fusion_model.onnx relative to this file.
        """
        import onnxruntime as ort

        if onnx_path is None:
            onnx_path = str(Path(__file__).parent / "data" / "fusion_model.onnx")

        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self._session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX session loaded: %s", onnx_path)

        # Lazy-loaded encoders
        self._deberta_tokenizer = None
        self._deberta_model = None
        self._clap_processor = None
        self._clap_model = None

    # ------------------------------------------------------------------
    # Lazy encoder loading
    # ------------------------------------------------------------------

    def _load_deberta(self):
        """Load DeBERTa-v3-base for text embedding. ~400 MB, first call only."""
        if self._deberta_model is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        model_name = "microsoft/deberta-v3-base"
        logger.info("Loading DeBERTa: %s", model_name)
        self._deberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._deberta_model = AutoModel.from_pretrained(model_name)
        self._deberta_model.eval()
        logger.info("DeBERTa loaded")

    def _load_clap(self):
        """Load CLAP for audio embedding. ~600 MB, first call only."""
        if self._clap_model is not None:
            return

        from transformers import ClapModel, ClapProcessor

        model_name = "laion/larger_clap_music"
        logger.info("Loading CLAP: %s", model_name)
        self._clap_processor = ClapProcessor.from_pretrained(model_name)
        self._clap_model = ClapModel.from_pretrained(model_name)
        self._clap_model.eval()
        logger.info("CLAP loaded")

    # ------------------------------------------------------------------
    # Embedding preparation (call once, reuse across batch)
    # ------------------------------------------------------------------

    def prepare_concept(self, concept_text: str) -> np.ndarray:
        """Encode concept text with DeBERTa. Call once, reuse across batch.

        Args:
            concept_text: The chromatic concept string (e.g. "RED temporal=Past
                spatial=Thing ontological=Known ...").

        Returns:
            768-dim embedding as float32 numpy array.
        """
        self._load_deberta()
        return self._encode_text(concept_text)

    def prepare_lyric(self, lyric_text: str) -> np.ndarray:
        """Encode lyric text with DeBERTa. Optional.

        Args:
            lyric_text: Lyric content for the segment.

        Returns:
            768-dim embedding as float32 numpy array.
        """
        self._load_deberta()
        return self._encode_text(lyric_text)

    def prepare_audio(self, waveform: np.ndarray, sr: int = 48000) -> np.ndarray:
        """Encode audio with CLAP. Optional — many candidates are MIDI-only.

        Args:
            waveform: Audio waveform as 1-D float32 numpy array.
            sr: Sample rate. CLAP expects 48000 Hz.

        Returns:
            512-dim embedding as float32 numpy array.
        """
        self._load_clap()
        return self._encode_audio(waveform, sr)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        midi_bytes: Optional[bytes] = None,
        audio_waveform: Optional[np.ndarray] = None,
        concept_text: Optional[str] = None,
        lyric_text: Optional[str] = None,
        concept_emb: Optional[np.ndarray] = None,
        audio_emb: Optional[np.ndarray] = None,
        lyric_emb: Optional[np.ndarray] = None,
    ) -> dict:
        """Score a single candidate.

        Provide either raw inputs (concept_text, midi_bytes, etc.) or
        precomputed embeddings (concept_emb, audio_emb, lyric_emb).
        Precomputed embeddings take priority when both are provided.

        At minimum, concept_text or concept_emb must be provided.

        Returns:
            {
                "temporal": {"past": 0.1, "present": 0.8, "future": 0.1},
                "spatial": {"thing": 0.05, "place": 0.9, "person": 0.05},
                "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
                "confidence": 0.89,
            }
        """
        # Build single-item batch
        candidate = {"midi_bytes": midi_bytes}
        if audio_waveform is not None:
            candidate["audio_waveform"] = audio_waveform
        if audio_emb is not None:
            candidate["audio_emb"] = audio_emb
        if lyric_text is not None:
            candidate["lyric_text"] = lyric_text
        if lyric_emb is not None:
            candidate["lyric_emb"] = lyric_emb

        # Prepare concept embedding
        if concept_emb is None:
            if concept_text is None:
                raise ValueError("Either concept_text or concept_emb is required")
            concept_emb = self.prepare_concept(concept_text)

        results = self.score_batch([candidate], concept_emb=concept_emb)
        return results[0]

    def score_batch(
        self,
        candidates: list[dict],
        concept_emb: Optional[np.ndarray] = None,
        concept_text: Optional[str] = None,
        lyric_emb: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """Score multiple candidates in one ONNX call.

        Each candidate dict may contain:
            - "midi_bytes": bytes (raw MIDI file content)
            - "audio_waveform": np.ndarray (1-D float32 waveform)
            - "audio_emb": np.ndarray (precomputed 512-dim CLAP embedding)
            - "lyric_text": str (lyric content)
            - "lyric_emb": np.ndarray (precomputed 768-dim DeBERTa embedding)

        concept_emb/concept_text apply to all candidates (same concept for
        the whole evolutionary batch). lyric_emb applies to all candidates
        unless overridden per-candidate.

        Returns:
            List of dicts sorted by confidence (descending), each containing:
            - "temporal", "spatial", "ontological": mode probability dicts
            - "confidence": float
            - "rank": int (0-based)
            - "candidate": the original candidate dict
        """
        if concept_emb is None:
            if concept_text is None:
                raise ValueError("Either concept_emb or concept_text is required")
            concept_emb = self.prepare_concept(concept_text)

        batch_size = len(candidates)
        if batch_size == 0:
            return []

        # Pre-allocate arrays
        piano_rolls = np.zeros((batch_size, 1, 128, 256), dtype=np.float32)
        audio_embs = np.zeros((batch_size, 512), dtype=np.float32)
        concept_embs = np.broadcast_to(
            concept_emb.reshape(1, 768), (batch_size, 768)
        ).copy()
        lyric_embs = np.zeros((batch_size, 768), dtype=np.float32)
        has_audio = np.zeros(batch_size, dtype=bool)
        has_midi = np.zeros(batch_size, dtype=bool)
        has_lyric = np.zeros(batch_size, dtype=bool)

        # Fill batch-level lyric embedding if provided
        if lyric_emb is not None:
            lyric_embs[:] = lyric_emb.reshape(1, 768)
            has_lyric[:] = True

        # Process each candidate
        for i, candidate in enumerate(candidates):
            # MIDI → piano roll
            midi = candidate.get("midi_bytes")
            if midi is not None and len(midi) > 0:
                try:
                    roll = midi_bytes_to_piano_roll(midi)
                    piano_rolls[i, 0] = roll
                    has_midi[i] = True
                except Exception as e:
                    logger.warning("Failed to convert MIDI for candidate %d: %s", i, e)

            # Audio embedding
            if "audio_emb" in candidate and candidate["audio_emb"] is not None:
                audio_embs[i] = candidate["audio_emb"]
                has_audio[i] = True
            elif (
                "audio_waveform" in candidate
                and candidate["audio_waveform"] is not None
            ):
                audio_embs[i] = self.prepare_audio(candidate["audio_waveform"])
                has_audio[i] = True

            # Lyric embedding (per-candidate override)
            if "lyric_emb" in candidate and candidate["lyric_emb"] is not None:
                lyric_embs[i] = candidate["lyric_emb"]
                has_lyric[i] = True
            elif "lyric_text" in candidate and candidate["lyric_text"] is not None:
                lyric_embs[i] = self.prepare_lyric(candidate["lyric_text"])
                has_lyric[i] = True

        # Run ONNX inference
        outputs = self._session.run(
            None,
            {
                "piano_roll": piano_rolls,
                "audio_emb": audio_embs,
                "concept_emb": concept_embs,
                "lyric_emb": lyric_embs,
                "has_audio": has_audio,
                "has_midi": has_midi,
                "has_lyric": has_lyric,
            },
        )
        temporal, spatial, ontological, confidence = outputs

        # Build result list
        results = []
        for i in range(batch_size):
            results.append(
                {
                    "temporal": {
                        mode.lower(): float(temporal[i, j])
                        for j, mode in enumerate(TEMPORAL_MODES)
                    },
                    "spatial": {
                        mode.lower(): float(spatial[i, j])
                        for j, mode in enumerate(SPATIAL_MODES)
                    },
                    "ontological": {
                        mode.lower(): float(ontological[i, j])
                        for j, mode in enumerate(ONTOLOGICAL_MODES)
                    },
                    "confidence": float(confidence[i, 0]),
                    "candidate": candidates[i],
                }
            )

        # Sort by confidence descending
        results.sort(key=lambda r: r["confidence"], reverse=True)
        for rank, result in enumerate(results):
            result["rank"] = rank

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text with DeBERTa, return mean-pooled 768-dim embedding."""
        import torch

        tokens = self._deberta_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            output = self._deberta_model(**tokens)
        # Mean pooling over token dimension
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        embeddings = output.last_hidden_state * attention_mask
        pooled = embeddings.sum(dim=1) / attention_mask.sum(dim=1)
        return pooled.squeeze(0).numpy().astype(np.float32)

    def _encode_audio(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Encode audio with CLAP, return 512-dim embedding."""
        import torch

        # CLAP expects 48kHz — resample if needed
        if sr != 48000:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=48000)

        # Use CLAP audio model directly (transformers 5.x compatible)
        inputs = self._clap_processor(
            audio=waveform,
            sampling_rate=48000,
            return_tensors="pt",
        )
        with torch.no_grad():
            audio_features = self._clap_model.audio_model(**inputs)
            audio_embeds = self._clap_model.audio_projection(audio_features[0][:, 0, :])

        return audio_embeds.squeeze(0).numpy().astype(np.float32)
