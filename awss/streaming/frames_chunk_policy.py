import librosa
import numpy as np

from awss.meta.streaming_interfaces import PolicyStates, ChunkPolicyInterface


class FramesChunkPolicy(ChunkPolicyInterface):
    def __init__(self, source_sr: int, models_sr: int, pacience: int = 4):
        self.source_sr = source_sr
        self.models_sr = models_sr
        self.pacience = pacience
        self.buffer_acumulated_for_asr = []
        self.state = PolicyStates.SKIP
        self.frames = None
        self.frames_accumulated = None

    def get_mode(self):
        return "audio_buffer"

    def process_audio_frames(self, audio_frames):
        if audio_frames == "close":
            return
        float64_buffer = np.frombuffer(audio_frames, dtype=np.int16) / 32767
        float64_buffer = librosa.core.resample(
            float64_buffer, self.source_sr, self.models_sr
        )
        self.buffer_acumulated_for_asr.append(float64_buffer)
        if len(self.buffer_acumulated_for_asr) > self.pacience:
            self.state = PolicyStates.CONSUME_PARTIAL_PREDS

    def callback_partial_preds(self, _):
        pass

    def consume_buffer(self):
        pass

    def consume_partial_preds(self):
        self.state = PolicyStates.SKIP
        return np.concatenate(self.buffer_acumulated_for_asr)

    def consume_final_prediction(self):
        self.state = PolicyStates.SKIP
        buffer = np.array([])
        if len(self.buffer_acumulated_for_asr) > 0:
            buffer = np.concatenate(self.buffer_acumulated_for_asr)
            self.buffer_acumulated_for_asr = []
        return buffer
