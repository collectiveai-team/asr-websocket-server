from abc import abstractmethod
from enum import Enum

import numpy as np


class ASRStreamingInterface:
    @abstractmethod
    def frames_to_logits(self, frames: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def logits_to_text(self, logits: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def frames_to_text(self, frames: np.ndarray, previous_transcript: str, **kwargs):
        pass


class VADModelInterface:
    @abstractmethod
    def user_is_speaking(self, buffer_frame: bytes):
        pass

    @abstractmethod
    def user_is_speaking_with_proba(self, buffer_frame: bytes) -> float:
        pass

    @abstractmethod
    def reset_states(self):
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
