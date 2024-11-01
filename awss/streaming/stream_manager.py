import sys
import logging
import threading
import traceback
from queue import Queue

from awss.meta.streaming_interfaces import (
    PolicyStates,
    VADModelInterface,
    ChunkPolicyInterface,
    ASRStreamingInterface,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamManager:
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
        self.vad_model = vad_model
        self.chunk_policy = chunk_policy
        self.source_sr = source_sr
        self.vad_sr = vad_sr
        self.asr_sr = asr_sr
        self.exit_event = threading.Event()
        self.running = True

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
        frames = b""
        consecutive_no_speech = 0
        consecutive_speech = 0
        n_frames_without_pause = 0
        self.n_frames = 0
        self.frames_information = [[]]
        logger.info(f"vad_sample rate => {self.vad_model.original_sr}")
        while not self.exit_event.is_set() and self.running:
            frame = stream_func()
            if frame == "close":
                self.asr_input_queue.put("pause_on_speech")
                break
            self.n_frames += 1
            is_speech = self.vad_model.user_is_speaking(frame)
            if is_speech:
                if consecutive_no_speech > 0:
                    self.frames_information[-1].append((consecutive_no_speech, 0))
                consecutive_no_speech = 0
                consecutive_speech += 1
                frames += frame
            else:
                if consecutive_speech > 0:
                    self.frames_information[-1].append((consecutive_speech, 1))
                if len(frames) > 10:
                    self.asr_input_queue.put(frames)
                frames = b""
                consecutive_no_speech += 1
                consecutive_speech = 0
            if consecutive_no_speech > 20 or (
                n_frames_without_pause > 80 and consecutive_no_speech > 5
            ):
                logger.info(
                    f"cconsecutive_no_speechon: {consecutive_no_speech}. n_frames_without_pause: {n_frames_without_pause} "
                )
                self.asr_input_queue.put("pause_on_speech")
                self.frames_information.append([])
                consecutive_no_speech = 0
                n_frames_without_pause = 0
            else:
                n_frames_without_pause = n_frames_without_pause + 1
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
        while not self.exit_event.is_set() and self.running:
            try:
                audio_frames = self.asr_input_queue.get()
                if audio_frames == "close":
                    break
                if audio_frames == "pause_on_speech":
                    final_preds = self.chunk_policy.consume_final_prediction()
                    if final_preds is None or len(final_preds) == 0:
                        continue
                    text = self.asr_model.frames_to_text(final_preds)

                    if text != "":
                        self.asr_output_queue.put(text)
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
