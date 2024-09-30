import os
import sys
import logging
import collections
from pathlib import Path

import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from pyctcdecode import build_ctcdecoder

from awss.meta.streaming_interfaces import ASRStreamingInterface

NEMO_CACHE = Path.joinpath(Path.home(), ".cache/torch/NeMo")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("nemo_logger").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


Hypotheses = collections.namedtuple(
    "Hypotheses",
    [
        "text",
        "state",
        "logits",
        "logit_score",
        "beam_score",
    ],
)


class ConformerCTCForStreaming(ASRStreamingInterface):
    def __init__(self, model_name, device: str = "cpu"):
        self.device = device

        if os.path.exists(model_name):
            self.model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
                model_name,
                map_location=self.device,
            )
        else:
            self.model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                model_name,
                map_location=self.device,
            )
        self.decoder = build_ctcdecoder(list(self.model.decoder.vocabulary))

        logger.info("Model loaded.")

    def frames_to_logits(self, audio_buffer: np.ndarray) -> np.ndarray:
        signal = torch.tensor(
            np.array([audio_buffer]),
            dtype=torch.float,
        ).to()
        input_signal_length = torch.tensor([len(audio_buffer)]).to(self.device)

        logits, *_ = self.model(
            input_signal=signal,
            input_signal_length=input_signal_length,
        )
        return logits.detach().numpy()[0]

    def logits_to_text(self, logits: np.ndarray) -> str:
        return self.decoder.decode(logits)

    def frames_to_text(self, audio_buffer: np.ndarray) -> str:
        logits = self.frames_to_logits(audio_buffer)
        return self.logits_to_text(logits)
