import logging

import torch
import whisper
import numpy as np

from awss.meta.streaming_interfaces import ASRStreamingInterface

logger = logging.getLogger(__name__)


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

    def frames_to_text(self, audio_buffer: np.ndarray) -> str:
        audio_buffer = audio_buffer.astype("float32")
        audio = whisper.pad_or_trim(audio_buffer)
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # decode the audio
        options = whisper.DecodingOptions(fp16=False, language="en")
        result = whisper.decode(self.model, mel, options)

        return result.text
