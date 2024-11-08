import logging
import re

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
        audio = whisper.pad_or_trim(audio_buffer)
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio, device=self.device).to(
            self.model.device
        )

        # decode the audio
        options = whisper.DecodingOptions(
            fp16=False, language="en", prompt=previous_transcript
        )
        result = whisper.decode(self.model, mel, options)

        transcript = result.text

        transcript = clean_repeated(transcript)

        return transcript
