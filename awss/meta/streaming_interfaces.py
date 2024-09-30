from enum import Enum
from abc import abstractmethod

import numpy as np


class ASRStreamingInterface:
    @abstractmethod
    def frames_to_logits(self, frames: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def logits_to_text(self, logits: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def frames_to_text(self, frames: np.ndarray, **kwargs):
        pass


class VADModelInterface:
    @abstractmethod
    def user_is_speaking(self, buffer_frame: bytes):
        pass


class PolicyStates(Enum):
    SKIP = 0
    CONSUME_PARTIAL_PREDS = 1
    CONSUME_FRAMES = 2
    CONSUME_FINAL_PREDICTION = 3


class ChunkPolicyInterface:
    @abstractmethod
    def process_audio_frames(self, audio_frames):
        pass

    @abstractmethod
    def callback_partial_preds(self, partial_preds):
        pass

    @abstractmethod
    def consume_buffer(self):
        pass

    @abstractmethod
    def consume_partial_preds(self):
        pass

    @abstractmethod
    def consume_final_prediction(self):
        pass

    @abstractmethod
    def get_mode(self):
        pass
