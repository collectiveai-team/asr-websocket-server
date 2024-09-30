import sys
import logging

import webrtcvad

from awss.meta.streaming_interfaces import VADModelInterface

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class WebRTCVAD(VADModelInterface):
    def __init__(self, intensity: int, original_sr: int):
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(intensity)
        self.original_sr = original_sr

    def user_is_speaking(self, buffer_frame: bytes):
        is_speech = True
        try:
            if len(buffer_frame) < 320:
                buffer_frame += b"\0" * (320 - len(buffer_frame))
            is_speech = self.vad.is_speech(buffer_frame, self.original_sr)
        except Exception as inst:
            logger.info(
                f"Vad exception {str(inst)}",
                inst,
                f"len of buffer_frame = {len(buffer_frame)}",
            )
        return is_speech
