import os
import sys
import threading
import traceback
from queue import Queue

import torch

from awss.logger import get_logger
from awss.meta.streaming_interfaces import (
    ASRStreamingInterface,
    ChunkPolicyInterface,
    PolicyStates,
    VADModelInterface,
)

logger = get_logger(__name__)


class SpeechFrameDetectorSilero:
    def __init__(self, vad_model, threshold: float = 0.5, sampling_rate: int = 16000):
        self.consecutive_no_speech = 0
        self.consecutive_speech = 0
        self.n_frames_without_pause = 0
        self.is_speech = False
        self.vad_model = vad_model

        # Silero specific parameters
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

        # Configurable thresholds
        self.min_silence_duration_ms = 100
        self.speech_pad_ms = 30
        self.min_silence_samples = sampling_rate * self.min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * self.speech_pad_ms / 1000

    def process_frame(self, frame):
        if not torch.is_tensor(frame):
            frame = torch.Tensor(frame)

        window_size_samples = len(frame[0]) if frame.dim() == 2 else len(frame)
        self.current_sample += window_size_samples

        speech_prob = self.vad_model(frame, self.sampling_rate).item()

        # Update speech state
        self.is_speech = speech_prob >= self.threshold

        if self.is_speech:
            self.handle_speech()
            if speech_prob >= self.threshold and self.temp_end:
                self.temp_end = 0
        else:
            self.handle_silence(speech_prob)

        self.n_frames_without_pause += 1

    def handle_speech(self):
        self.consecutive_no_speech = 0
        self.consecutive_speech += 1

        if not self.triggered:
            self.triggered = True

    def handle_silence(self, speech_prob):
        self.consecutive_speech = 0
        self.consecutive_no_speech += 1

        if speech_prob < (self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample

    def should_pause(self):
        if not self.triggered:
            return False

        if self.temp_end and (
            self.current_sample - self.temp_end >= self.min_silence_samples
        ):
            self.triggered = False
            return True

        return False

    def reset_pause_counters(self):
        self.consecutive_no_speech = 0
        self.n_frames_without_pause = 0
        self.temp_end = 0
        self.current_sample = 0
        self.triggered = False
        self.vad_model.reset_states()


class SpeechFrameDetectorWebRTC:
    def __init__(self, vad_model):
        self.consecutive_no_speech = 0
        self.consecutive_speech = 0
        self.n_frames_without_pause = 0
        self.is_speech = False
        self.vad_model = vad_model
        self.n_frames_without_pause_threshold = int(
            os.getenv("N_FRAMES_WITHOUT_PAUSE_THRESHOLD", "80")
        )
        self.consecutive_no_speech_lower_threshold = int(
            os.getenv("CONSECUTIVE_NO_SPEECH_LOWER_THRESHOLD", "3")
        )
        self.consecutive_no_speech_upper_threshold = int(
            os.getenv("CONSECUTIVE_NO_SPEECH_UPPER_THRESHOLD", "20")
        )

    def process_frame(self, frame):
        self.is_speech = self.vad_model.user_is_speaking(frame)
        if self.is_speech:
            self.handle_speech()
        else:
            self.handle_silence()
        self.n_frames_without_pause += 1

    def handle_speech(self):
        self.consecutive_no_speech = 0
        self.consecutive_speech += 1

    def handle_silence(self):
        self.consecutive_speech = 0
        self.consecutive_no_speech += 1

    def should_pause(self):
        return (
            # self.consecutive_no_speech > self.consecutive_no_speech_upper_threshold
            False
            or (
                self.n_frames_without_pause > self.n_frames_without_pause_threshold
                and self.consecutive_no_speech
                > self.consecutive_no_speech_lower_threshold
            )
        )

    def reset_pause_counters(self):
        self.consecutive_no_speech = 0
        self.n_frames_without_pause = 0


class FrameAccumulator:
    def __init__(self, exclude_silences: bool = True):
        self.frames = b""
        self.exclude_silences = exclude_silences
        self.frames_information = [[]]

    def add_frame(self, frame, detector: SpeechFrameDetectorWebRTC):
        if detector.is_speech:
            self.frames += frame
            if detector.consecutive_no_speech > 0:
                self.frames_information[-1].append((detector.consecutive_no_speech, 0))
        else:
            if detector.consecutive_no_speech < 20 or not self.exclude_silences:
                self.frames += frame

            if detector.consecutive_speech > 0:
                self.frames_information[-1].append((detector.consecutive_speech, 1))

    def should_process(self):
        return len(self.frames) > 10

    def reset(self):
        self.frames = b""
        self.frames_information.append([])


class StreamManager:
    def __init__(
        self,
        asr_model: ASRStreamingInterface,
        vad_model: VADModelInterface,
        chunk_policy: ChunkPolicyInterface,
        source_sr: int = 48_000,
        vad_sr: int = 16_000,
        asr_sr: int = 16_000,
        exclude_silences: bool = True,
    ):
        self.asr_model = asr_model
        self.vad_model = vad_model
        self.chunk_policy = chunk_policy
        self.source_sr = source_sr
        self.vad_sr = vad_sr
        self.asr_sr = asr_sr
        self.exit_event = threading.Event()
        self.running = True
        self.exclude_silences = exclude_silences

    def stop(self):
        """stop the asr process"""
        self.exit_event.set()
        self.asr_input_queue.put("close")
        self.running = False
        self.asr_output_queue.put("close")
        self.asr_output_queue.task_done()
        self.asr_input_queue.task_done()
        print("asr stopped")

    def start(self, stream_func):
        """start the asr process"""
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()
        self.asr_process = threading.Thread(target=self.asr_process)
        self.asr_process.start()
        self.vad_process = threading.Thread(
            target=self.vad_process,
            args=[stream_func],
        )
        self.vad_process.start()

    def vad_process(self, stream_func):
        speech_detector = SpeechFrameDetectorSilero(self.vad_model)
        frame_accumulator = FrameAccumulator(exclude_silences=self.exclude_silences)

        while (
            not self.exit_event.is_set() and self.running
        ):  # Using the existing self.running attribute
            frame = stream_func()
            if frame == "close":
                self.asr_input_queue.put("pause_on_speech")
                break

            speech_detector.process_frame(frame)
            frame_accumulator.add_frame(frame, speech_detector)

            if frame_accumulator.should_process():
                self.asr_input_queue.put(frame_accumulator.frames)
                frame_accumulator.reset()

            if speech_detector.should_pause():
                logger.info(
                    f"consecutive_no_speech: {speech_detector.consecutive_no_speech}. n_frames_without_pause: {speech_detector.n_frames_without_pause}"
                )
                self.asr_input_queue.put("pause_on_speech")
                frame_accumulator.reset()
                speech_detector.reset_pause_counters()

        logger.info("Ending vad_process!")

    def update_params(self, source_sr: int, vad_sr: int):
        if source_sr:
            self.source_sr = source_sr
            self.vad_sr = vad_sr
            self.chunk_policy.source_sr = source_sr
        if vad_sr:
            self.vad_model.original_sr = vad_sr

    def asr_process(self):

        logger.info("\nlistening...\n")
        previous_transcript = ""
        while not self.exit_event.is_set() and self.running:
            try:
                audio_frames = self.asr_input_queue.get()
                if audio_frames == "close":
                    break
                if audio_frames == "pause_on_speech":
                    final_preds = self.chunk_policy.consume_final_prediction()
                    if final_preds is None or len(final_preds) == 0:
                        continue
                    text = self.asr_model.frames_to_text(
                        final_preds, previous_transcript=previous_transcript
                    )

                    if text != "":
                        self.asr_output_queue.put(text)
                        previous_transcript += text
                        previous_transcript = (
                            ".".join(previous_transcript.split(".")[-2:])
                            if "." in previous_transcript
                            else previous_transcript
                        )
                        text = None
                    continue
                self.chunk_policy.process_audio_frames(audio_frames)
                if self.chunk_policy.state == PolicyStates.CONSUME_PARTIAL_PREDS:
                    partial_preds = self.chunk_policy.consume_partial_preds()
                    if len(partial_preds) == 0:
                        continue
                    text = self.asr_model.frames_to_text(partial_preds)
                    if text != "":
                        self.asr_output_queue.put(text)
                        text = None
            except Exception as inst:
                print("Exception in thread asr => ", inst)
                trace = traceback.format_stack()[:-1]
                trace.extend(traceback.format_tb(sys.exc_info()[2]))
                trace.extend(
                    traceback.format_exception_only(
                        sys.exc_info()[0], sys.exc_info()[1]
                    )
                )
                trc = "Traceback (most recent call last):\n"
                logger.error(f"{trc}{''.join(trace)}{trc}")
                continue
        logger.info("Ending asr_process!")

    def get_last_text(self):
        """returns the text, sample length and inference time in seconds."""
        return self.asr_output_queue.get()
