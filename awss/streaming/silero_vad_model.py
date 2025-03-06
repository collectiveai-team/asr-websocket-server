import numpy as np
import torch

from awss.logger import get_logger
from awss.meta.streaming_interfaces import VADModelInterface

logger = get_logger(__name__)


class SileroVAD(VADModelInterface):
    def __init__(self, intensity: int, original_sr: int):
        logger.info(
            f"Initializing SileroVAD model, with intensity {intensity}, sr {original_sr}"
        )
        self.original_sr = original_sr

        # Map intensity (0-3) to threshold values
        threshold_map = {
            0: 0.8,  # More aggressive
            1: 0.7,
            2: 0.6,
            3: 0.5,  # Less aggressive
        }
        self.threshold = threshold_map.get(intensity, 0.8)

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
        )
        self.model.eval()
        self.model = self.model.to(torch.float32)  # Set model to float32
        self.model = (
            self.model.to("cuda") if torch.cuda.is_available() else self.model
        )  # Move model to GPU if available

        logger.info(
            "Loaded SileroVAD model. Running in {} mode".format(
                "GPU" if torch.cuda.is_available() else "CPU"
            )
        )

    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)

    def user_is_speaking_with_proba(self, buffer_frame: bytes) -> float:
        speech_prob = 0.5
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(buffer_frame, dtype=np.int16)

            # Normalize audio to float between -1 and 1
            audio_float = audio_array.astype(np.float32) / 32768.0

            # Determine required padding size based on sample rate
            window_size = 512 if self.original_sr == 16000 else 256

            # Used for ONNX implementation
            # context_size = 64 if self.original_sr == 16000 else 32

            # Pad array if needed
            if len(audio_float) < window_size:
                padding = np.zeros(window_size - len(audio_float))
                audio_float = np.concatenate([audio_float, padding])

            # Reshape for model input (add batch dimension)
            audio_float = audio_float.reshape(1, -1)
            audio_tensor = torch.from_numpy(audio_float).to(
                torch.float32
            )  # Explicitly use float32

            audio_tensor = audio_tensor.to(
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # Convert to torch tensor
            # Convert to torch tensor with explicit dtype
            # audio_tensor = torch.from_numpy(audio_float).to(torch.float64)

            # Get speech probability
            speech_prob = self.model(audio_tensor, self.original_sr).item()
        except Exception as e:
            logger.error(f"Silero VAD exception: {str(e)}")
        return speech_prob

    def user_is_speaking(self, buffer_frame: bytes) -> bool:

        speech_prob = self.user_is_speaking_with_proba(buffer_frame=buffer_frame)
        return speech_prob > self.threshold

    def reset_states(self):
        self.model.reset_states()
