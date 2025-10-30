import logging
import os
import assemblyai as aai
from dotenv import load_dotenv

from app.agents.enums.chain_artifact_file_type import ChainArtifactFileType
from app.agents.models.audio_chain_artifact_file import AudioChainArtifactFile
from app.agents.models.text_chain_artifact_file import TextChainArtifactFile
from app.agents.tools.text_tools import save_artifact_file_to_md

load_dotenv()


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
        logging.error("ASSEMBLYAI_API_KEY not found in environment variables!")
        return None

    aai.settings.api_key = api_key
    full_path = f"{working_path}/{file_name}"

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

    logging.info("Starting aggressive AssemblyAI transcription...")
    transcriber = aai.Transcriber(config=aai_config)
    evp_transcription = transcriber.transcribe(full_path)

    if evp_transcription.status == 'error':
        logging.error(f"Transcription failed: {evp_transcription.error}")
        raise RuntimeError(f"Transcription failed: {evp_transcription.error}")

    if evp_transcription.status == 'completed':
        logging.info(f"Transcription completed!")
        logging.info(f"Text: {evp_transcription.text}")
        logging.info(f"Confidence: {evp_transcription.confidence}")
        if hasattr(evp_transcription, 'sentiment_analysis_results') and evp_transcription.sentiment_analysis_results:
            logging.info(f"Sentiment: {evp_transcription.sentiment_analysis_results}")
        if hasattr(evp_transcription, 'entities') and evp_transcription.entities:
            logging.info(f"Detected entities: {[e.text for e in evp_transcription.entities]}")
        if hasattr(evp_transcription, 'summary') and evp_transcription.summary:
            logging.info(f"Summary: {evp_transcription.summary}")
        if hasattr(evp_transcription, 'utterances') and evp_transcription.utterances:
            logging.info(f"Utterances: {len(evp_transcription.utterances)} detected")
            for utterance in evp_transcription.utterances[:3]:  # First 3
                logging.info(f"  Utterance: '{utterance.text}' (confidence: {utterance.confidence})")
        if evp_transcription.text and evp_transcription.text.strip():
            return evp_transcription.text
        elif hasattr(evp_transcription, 'utterances') and evp_transcription.utterances:
            utterance_texts = [u.text for u in evp_transcription.utterances if u.text.strip()]
            if utterance_texts:
                combined_text = " ".join(utterance_texts)
                logging.info(f"Using combined utterances: {combined_text}")
                return combined_text
    logging.warning("No transcription text generated - AssemblyAI too conservative!")
    return None


def chain_artifact_file_from_speech_to_text(audio: AudioChainArtifactFile,
                                            thread_id: str) -> TextChainArtifactFile | None:


    transcript_text = evp_speech_to_text(audio.get_artifact_path(with_file_name=False), audio.file_name)
    if transcript_text:
        transcript = TextChainArtifactFile(
            text_content=transcript_text,
            thread_id=thread_id,
            rainbow_color=audio.rainbow_color,
            base_path=audio.base_path,
            chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
            artifact_name=f"{audio.artifact_name}_transcript",
        )
        return transcript
    else:
        logging.warning("No transcript generated from audio.")
        return None
