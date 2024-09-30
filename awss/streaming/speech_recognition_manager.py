import io
import sys
import wave
import logging
import threading
from queue import Queue

import speech_recognition as sr

from awss.meta.streaming_interfaces import (
    VADModelInterface,
    ChunkPolicyInterface,
    ASRStreamingInterface,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechRecognitionStreamManager:
    def __init__(
        self,
        asr_model: ASRStreamingInterface,
        vad_model: VADModelInterface,
        chunk_policy: ChunkPolicyInterface,
        source_sr: int = 48_000,
        vad_sr: int = 16_000,
        asr_sr: int = 16_000,
    ):
        self.asr_model = asr_model
        self.chunk_policy = chunk_policy
        self.source_sr = source_sr
        self.vad_sr = vad_sr
        self.asr_sr = asr_sr
        self.exit_event = threading.Event()
        self.running = True
        self.recognizer = sr.Recognizer()
        self.audio_buffer = b""
        self.buffer_size = int(self.source_sr * 0.5)  # 0.5 seconds of audio

    def stop(self):
        self.exit_event.set()
        self.running = False
        self.asr_output_queue.put("close")
        self.asr_output_queue.task_done()
        print("asr stopped")

    def start(self, stream_func):
        self.asr_output_queue = Queue()
        self.asr_process = threading.Thread(
            target=self.asr_process, args=(stream_func,)
        )
        self.asr_process.start()

    def asr_process(self, stream_func):
        logger.info("\nlistening...\n")

        # Step 1: Collect Ambient Noise Buffer
        ambient_buffer = b""
        for _ in range(50):  # Collect ~1 second of ambient noise (assuming 50 chunks)
            ambient_buffer += stream_func()

        # Step 2: Convert Raw Audio to WAV Format in Memory
        # Create an in-memory bytes buffer
        wav_io = io.BytesIO()

        # Open the in-memory buffer as a WAV file
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(1)  # Set number of channels (1 for mono, 2 for stereo)
            wf.setsampwidth(2)  # Set sample width in bytes (2 for 16-bit audio)
            wf.setframerate(self.source_sr)  # Set the sample rate
            wf.writeframes(ambient_buffer)  # Write the raw audio data

        # Important: Seek to the beginning of the BytesIO buffer so it can be read from the start
        wav_io.seek(0)

        # Step 3: Create an AudioFile Instance from the WAV Buffer
        ambient_source = sr.AudioFile(wav_io)

        # Step 4: Adjust for Ambient Noise
        with ambient_source as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

        while not self.exit_event.is_set() and self.running:
            try:
                frame = stream_func()
                if frame == "close":
                    break

                self.audio_buffer += frame

                if len(self.audio_buffer) >= self.buffer_size:
                    audio_data = sr.AudioData(self.audio_buffer, self.source_sr, 2)
                    self.audio_buffer = b""

                    text = self.recognizer.recognize_whisper(
                        audio_data,
                    )
                    if text:
                        self.asr_output_queue.put(text)
            except sr.UnknownValueError:
                logger.info("Speech not recognized")
            except Exception as e:
                logger.error(f"Error in speech recognition: {e}")

        logger.info("Ending asr_process!")

    def update_params(self, source_sr: int, vad_sr: int):
        if source_sr:
            self.source_sr = source_sr
            self.buffer_size = int(self.source_sr * 0.5)
        if vad_sr:
            self.vad_sr = vad_sr

    def get_last_text(self):
        return self.asr_output_queue.get()
