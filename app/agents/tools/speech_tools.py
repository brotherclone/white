import logging
import os
import tempfile

import assemblyai as aai
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def evp_speech_to_text(working_path: str, file_name: str) -> str | None:
    """
    Use AssemblyAI to transcribe audio with aggressive settings.
    This is intended for very noisy audio where other services fail.
    It may hallucinate text, so use with caution.

    :param working_path:
    :param file_name:
    :return:
    """
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        logger.error("ASSEMBLYAI_API_KEY not found in environment variables!")
        return None

    aai.settings.api_key = api_key
    full_path = f"{working_path}/{file_name}"

    if not os.path.exists(full_path):
        logger.error(f"Audio file not found: {full_path}")
        return None

    aai_config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.universal,
        filter_profanity=False,
        format_text=False,  # Don't clean up the output
        punctuate=False,  # Don't add smart punctuation
        language_code=None,  # Let it auto-detect (might get it wrong)
        language_detection=True,
        disfluencies=True,  # Include "uh", "um", etc.
        speaker_labels=False,
        speakers_expected=None,  # Let it guess
        sentiment_analysis=False,  # Might hallucinate emotions
        entity_detection=True,  # Might hallucinate entities
    )

    logger.info("Starting aggressive AssemblyAI transcription...")
    transcriber = aai.Transcriber(config=aai_config)
    try:
        evp_transcription = transcriber.transcribe(full_path)
    except Exception as e:
        logger.error(f"AssemblyAI transcription failed with exception: {e}")
        return None
    if evp_transcription.status == "error":
        logger.error(f"Transcription failed: {evp_transcription.error}")
        return None

    if evp_transcription.status == "completed":
        logger.info("Transcription completed!")
        logger.info(f"Text: {evp_transcription.text}")
        logger.info(f"Confidence: {evp_transcription.confidence}")
        if (
            hasattr(evp_transcription, "sentiment_analysis_results")
            and evp_transcription.sentiment_analysis_results
        ):
            logger.info(f"Sentiment: {evp_transcription.sentiment_analysis_results}")
        if hasattr(evp_transcription, "entities") and evp_transcription.entities:
            logger.info(
                f"Detected entities: {[e.text for e in evp_transcription.entities]}"
            )
        if hasattr(evp_transcription, "summary") and evp_transcription.summary:
            logger.info(f"Summary: {evp_transcription.summary}")
        if hasattr(evp_transcription, "utterances") and evp_transcription.utterances:
            logger.info(f"Utterances: {len(evp_transcription.utterances)} detected")
            for utterance in evp_transcription.utterances[:3]:  # First 3
                logger.info(
                    f"  Utterance: '{utterance.text}' (confidence: {utterance.confidence})"
                )
        if evp_transcription.text and evp_transcription.text.strip():
            return evp_transcription.text
        elif hasattr(evp_transcription, "utterances") and evp_transcription.utterances:
            utterance_texts = [
                u.text for u in evp_transcription.utterances if u.text.strip()
            ]
            if utterance_texts:
                combined_text = " ".join(utterance_texts)
                logger.info(f"Using combined utterances: {combined_text}")
                return combined_text

    logger.warning("No transcription text generated - AssemblyAI too conservative!")
    return None


def transcription_from_speech_to_text(audio_data: np.ndarray, sample_rate: int) -> str:
    """Transcribe in-memory audio data via AssemblyAI.

    Writes audio to a temporary WAV file for the API, then cleans up.
    """
    block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, audio_data, sample_rate, subtype="PCM_16")
    try:
        working_path = os.path.dirname(tmp_path)
        file_name = os.path.basename(tmp_path)
        transcript_text = evp_speech_to_text(working_path, file_name)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    if transcript_text:
        logger.info(f"✓ Generated transcript with {len(transcript_text)} characters")
        return transcript_text
    else:
        if block_mode:
            raise Exception("No transcript generated - aborting workflow")
        logger.warning("⚠️  No transcript generated - using placeholder")
    return "[EVP: No discernible speech detected]"
