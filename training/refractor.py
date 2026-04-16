"""
Refractor — fitness function for evolutionary music composition.

Wraps the ONNX-exported multimodal fusion model with lazy-loaded DeBERTa
and CLAP encoders. Designed for CPU inference with batch scoring of 50+
candidates per evolutionary stage.

Usage:
    scorer = Refractor()

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

# Import midi_bytes_to_piano_roll without triggering training.models.__init__
# (which eagerly imports torch-dependent classifiers)
import importlib.util as _ilu  # noqa: E402
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.structures.concepts.chromatic_targets import (
    CHROMATIC_TARGETS as _CDM_CHROMATIC_TARGETS,
)
from app.structures.concepts.chromatic_targets import (
    ONTOLOGICAL_MODES,
    SPATIAL_MODES,
    TEMPORAL_MODES,
)

_spec = _ilu.spec_from_file_location(
    "piano_roll_encoder",
    str(Path(__file__).parent / "models" / "piano_roll_encoder.py"),
)
_pre = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pre)
midi_bytes_to_piano_roll = _pre.midi_bytes_to_piano_roll

logger = logging.getLogger(__name__)


# Canonical color order for the CDM classification head (matches modal_train_refractor_cdm.py)
_CDM_COLOR_ORDER = [
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Blue",
    "Indigo",
    "Violet",
    "White",
    "Black",
]


class Refractor:
    """Scores audio/MIDI candidates against chromatic concepts.

    The ONNX model (16 MB) loads immediately. DeBERTa (~400 MB) and CLAP
    (~600 MB) are lazy-loaded on first use. For MIDI-only scoring (the
    primary evolutionary use case), CLAP never loads.
    """

    def __init__(
        self,
        onnx_path: Optional[str] = None,
        cdm_onnx_path: Optional[str] = None,
    ):
        """Initialize the scorer with an ONNX model.

        Args:
            onnx_path: Path to refractor.onnx. Defaults to
                training/data/refractor.onnx relative to this file.
            cdm_onnx_path: Optional path to refractor_cdm.onnx. When provided,
                audio-only score() calls are routed through the CDM calibration
                head instead of the base model. Pass an empty string to disable
                auto-detection. Defaults to auto-detecting
                training/data/refractor_cdm.onnx if it exists.
        """
        import onnxruntime as ort

        if onnx_path is None:
            onnx_path = str(Path(__file__).parent / "data" / "refractor.onnx")

        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self._session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        logger.info("ONNX session loaded: %s", onnx_path)

        # Refractor CDM — optional calibration head for full-mix audio scoring
        self._cdm_session = None
        self._cdm_onnx_path: Optional[str] = None

        if cdm_onnx_path is None:
            # Auto-detect
            default_cdm = Path(__file__).parent / "data" / "refractor_cdm.onnx"
            if default_cdm.exists():
                cdm_onnx_path = str(default_cdm)
        elif cdm_onnx_path == "":
            cdm_onnx_path = None  # Explicitly disabled

        if cdm_onnx_path and Path(cdm_onnx_path).exists():
            self._cdm_session = ort.InferenceSession(
                cdm_onnx_path,
                providers=["CPUExecutionProvider"],
            )
            self._cdm_onnx_path = cdm_onnx_path
            logger.info("Refractor CDM session loaded: %s", cdm_onnx_path)

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

    def prepare_sounds_like(self, artist_descriptions: list[str]) -> np.ndarray:
        """Embed a list of artist description strings, mean-pool to 768-dim.

        Args:
            artist_descriptions: List of aesthetic description strings (one per artist).

        Returns:
            768-dim float32 numpy array (zero vector if list is empty).
        """
        if not artist_descriptions:
            return np.zeros(768, dtype=np.float32)

        self._load_deberta()
        embeddings = [self._encode_text(desc) for desc in artist_descriptions]
        return np.mean(embeddings, axis=0).astype(np.float32)

    def score(
        self,
        midi_bytes: Optional[bytes] = None,
        audio_waveform: Optional[np.ndarray] = None,
        concept_text: Optional[str] = None,
        lyric_text: Optional[str] = None,
        concept_emb: Optional[np.ndarray] = None,
        audio_emb: Optional[np.ndarray] = None,
        lyric_emb: Optional[np.ndarray] = None,
        sounds_like_texts: Optional[list[str]] = None,
        sounds_like_emb: Optional[np.ndarray] = None,
    ) -> dict:
        """Score a single candidate.

        Provide either raw inputs (concept_text, midi_bytes, etc.) or
        precomputed embeddings (concept_emb, audio_emb, lyric_emb).
        Precomputed embeddings take priority when both are provided.

        At minimum, concept_text or concept_emb must be provided.

        sounds_like_texts: list of artist description strings to embed on-the-fly.
        sounds_like_emb: pre-computed 768-dim embedding (takes priority).
        When neither is provided the null path is used (backward compatible).

        Returns:
            {
                "temporal": {"past": 0.1, "present": 0.8, "future": 0.1},
                "spatial": {"thing": 0.05, "place": 0.9, "person": 0.05},
                "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
                "confidence": 0.89,
            }
        """
        # --- Refractor CDM routing ---
        # When the CDM is loaded and this is an audio-only call (no MIDI),
        # route through the calibration head for full-mix scoring.
        if (
            self._cdm_session is not None
            and midi_bytes is None
            and audio_waveform is None
            and audio_emb is not None
        ):
            if concept_emb is None:
                if concept_text is None:
                    raise ValueError("Either concept_text or concept_emb is required")
                concept_emb = self.prepare_concept(concept_text)
            return self._score_cdm(audio_emb, concept_emb)

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

        # Resolve sounds_like embedding
        resolved_sl_emb = None
        if sounds_like_emb is not None:
            resolved_sl_emb = sounds_like_emb
        elif sounds_like_texts is not None:
            resolved_sl_emb = self.prepare_sounds_like(sounds_like_texts)

        results = self.score_batch(
            [candidate],
            concept_emb=concept_emb,
            sounds_like_emb=resolved_sl_emb,
        )
        return results[0]

    def score_batch(
        self,
        candidates: list[dict],
        concept_emb: Optional[np.ndarray] = None,
        concept_text: Optional[str] = None,
        lyric_emb: Optional[np.ndarray] = None,
        sounds_like_emb: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """Score multiple candidates in one ONNX call.

        Each candidate dict may contain:
            - "midi_bytes": bytes (raw MIDI file content)
            - "audio_waveform": np.ndarray (1-D float32 waveform)
            - "audio_emb": np.ndarray (precomputed 512-dim CLAP embedding)
            - "lyric_text": str (lyric content)
            - "lyric_emb": np.ndarray (precomputed 768-dim DeBERTa embedding)

        concept_emb/concept_text apply to all candidates (same concept for
        the whole evolutionary batch). lyric_emb and sounds_like_emb apply
        to all candidates unless overridden per-candidate.

        sounds_like_emb: optional 768-dim embedding broadcast across the batch.
            When None, the null path is used (has_sounds_like=False for all).

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
        sounds_like_embs = np.zeros((batch_size, 768), dtype=np.float32)
        has_audio = np.zeros(batch_size, dtype=bool)
        has_midi = np.zeros(batch_size, dtype=bool)
        has_lyric = np.zeros(batch_size, dtype=bool)
        has_sounds_like = np.zeros(batch_size, dtype=bool)

        # Fill batch-level lyric embedding if provided
        if lyric_emb is not None:
            lyric_embs[:] = lyric_emb.reshape(1, 768)
            has_lyric[:] = True

        # Fill batch-level sounds_like embedding if provided
        if sounds_like_emb is not None:
            sounds_like_embs[:] = sounds_like_emb.reshape(1, 768)
            has_sounds_like[:] = True

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

        # Detect whether ONNX model supports sounds_like inputs
        input_names = {inp.name for inp in self._session.get_inputs()}
        use_sounds_like = "sounds_like_emb" in input_names

        feed = {
            "piano_roll": piano_rolls,
            "audio_emb": audio_embs,
            "concept_emb": concept_embs,
            "lyric_emb": lyric_embs,
            "has_audio": has_audio,
            "has_midi": has_midi,
            "has_lyric": has_lyric,
        }
        if use_sounds_like:
            feed["sounds_like_emb"] = sounds_like_embs
            feed["has_sounds_like"] = has_sounds_like

        # Run ONNX inference
        outputs = self._session.run(None, feed)
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

    def _score_cdm(
        self,
        audio_emb: np.ndarray,
        concept_emb: np.ndarray,
    ) -> dict:
        """Score a single audio embedding via the Refractor CDM head.

        The CDM ONNX model outputs color_probs (batch, 9) — a softmax over the
        canonical _CDM_COLOR_ORDER. The top predicted color is mapped to its
        CHROMATIC_TARGETS distributions, which are returned in the same dict
        shape as score_batch().

        Confidence = max color probability (how decisive the model is).
        """
        # CDM ONNX input: concatenated [clap_emb, concept_emb] = (1, 1280)
        x = (
            np.concatenate([audio_emb, concept_emb], axis=-1)
            .reshape(1, -1)
            .astype(np.float32)
        )
        outputs = self._cdm_session.run(None, {"input": x})
        color_probs = outputs[0][0]  # (9,) float32

        best_idx = int(np.argmax(color_probs))
        best_color = _CDM_COLOR_ORDER[best_idx]
        confidence = float(color_probs[best_idx])

        targets = _CDM_CHROMATIC_TARGETS.get(
            best_color, _CDM_CHROMATIC_TARGETS["White"]
        )
        temporal = {
            m.lower(): float(targets["temporal"][j])
            for j, m in enumerate(TEMPORAL_MODES)
        }
        spatial = {
            m.lower(): float(targets["spatial"][j]) for j, m in enumerate(SPATIAL_MODES)
        }
        ontological = {
            m.lower(): float(targets["ontological"][j])
            for j, m in enumerate(ONTOLOGICAL_MODES)
        }

        return {
            "temporal": temporal,
            "spatial": spatial,
            "ontological": ontological,
            "confidence": confidence,
            "predicted_color": best_color,  # direct CDM argmax — bypass distribution round-trip
            "candidate": {},
            "rank": 0,
        }

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
        return np.array(pooled.squeeze(0).tolist(), dtype=np.float32)

    def _encode_audio(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Encode audio with CLAP, return 512-dim embedding."""
        import torch

        # CLAP expects 48kHz — resample if needed
        if sr != 48000:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=48000)

        raw = self._clap_processor(
            audios=[waveform],
            sampling_rate=48000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            "input_features": raw["input_features"].float(),
            "is_longer": raw["is_longer"],
        }
        with torch.no_grad():
            # get_audio_features handles pooling + projection internally.
            # For long files CLAP windows the audio; mean-pool across windows.
            audio_embeds = self._clap_model.get_audio_features(**inputs)

        # audio_embeds: (num_windows, 512) — mean-pool to (512,)
        return np.array(audio_embeds.mean(dim=0).tolist(), dtype=np.float32)
