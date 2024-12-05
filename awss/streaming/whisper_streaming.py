import logging
import re
import wave

import numpy as np
import torch
import whisper

from awss.meta.streaming_interfaces import ASRStreamingInterface

logger = logging.getLogger(__name__)


def clean_repeated(text: str) -> str:
    pattern = r"(\b.+?\b)(\s*\1\s*)+"

    matches = re.finditer(pattern, text)
    for match in matches:
        text = text.replace(match.group(), "")
    return text


class WhisperForStreaming(ASRStreamingInterface):
    def __init__(self, model_name="base", device: str = None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = whisper.load_model("base.en")
        logger.info("Model loaded.")

    def frames_to_logits(self, audio_buffer: np.ndarray) -> np.ndarray:
        raise Exception("Not implemented.")

    def logits_to_text(self, logits: np.ndarray) -> str:
        raise Exception("Not implemented.")

    def frames_to_text(
        self, audio_buffer: np.ndarray, previous_transcript: str = ""
    ) -> str:
        audio_buffer = audio_buffer.astype("float32")
        resutl = self.model.transcribe(audio=audio_buffer)

        transcript = resutl["text"]

        transcript = clean_repeated(transcript)

        return transcript

    def dump_audio_chunk(audio_buffer: np.ndarray, transcript: str):
        with open(f"/resources/{transcript.replace(' ', '_')}.wav", "wb") as temp_wav:
            with wave.open(temp_wav.name, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono audio
                wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                wav_file.setframerate(16000)  # Sample rate
                # Normalize audio data to prevent noise/clipping
                normalized_audio = np.int16(audio_buffer * 32767)
                wav_file.writeframes(normalized_audio.tobytes())
