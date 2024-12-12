import os
import math

import torch


class SpeechFrameDetector:
    def __init__(
        self,
        vad_model,
        threshold: float = 0.5,
        neg_threshold: float = None,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = math.inf,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        window_size_samples: int = 512,
    ):
        """
        A detector that emulates get_speech_timestamps logic in a streaming fashion, but processes one frame at a time.

        Parameters
        ----------
        vad_model:
            A preloaded silero VAD model (or similar) that given a frame and sampling rate, returns speech probability.

        threshold: float
            Speech threshold. Probabilities above this are considered speech.

        neg_threshold: float
            Negative threshold (for ending speech). If None, defaults to threshold - 0.15.

        sampling_rate: int
            Audio sampling rate (typically 16000).

        min_speech_duration_ms: int
            Minimum duration of a valid speech segment in ms.

        max_speech_duration_s: float
            Maximum duration of a speech segment. If exceeded, segment is forced to end.

        min_silence_duration_ms: int
            Minimum silence needed to consider speech ended.

        speech_pad_ms: int
            Padding around speech segments (not directly needed for pause decision, but included for completeness).

        window_size_samples: int
            Number of samples per frame.
        """

        min_speech_duration_ms = os.getenv(
            "MIN_SPEECH_DURATION_MS", min_speech_duration_ms
        )
        max_speech_duration_s = os.getenv(
            "MAX_SPEECH_DURATION_S", max_speech_duration_s
        )
        min_silence_duration_ms = os.getenv(
            "MIN_SILENCE_DURATION_MS", min_silence_duration_ms
        )

        self.vad_model = vad_model
        self.threshold = threshold
        self.neg_threshold = (
            neg_threshold if neg_threshold is not None else max(threshold - 0.15, 0.01)
        )
        self.sampling_rate = sampling_rate
        self.window_size_samples = window_size_samples

        self.min_speech_samples = self._ms_to_samples(min_speech_duration_ms)
        self.min_silence_samples = self._ms_to_samples(min_silence_duration_ms)
        self.speech_pad_samples = self._ms_to_samples(speech_pad_ms)

        if math.isinf(max_speech_duration_s):
            self.max_speech_samples = math.inf
        else:
            self.max_speech_samples = (
                (max_speech_duration_s * self.sampling_rate)
                - self.window_size_samples
                - 2 * self.speech_pad_samples
            )

        # State variables
        self.triggered = False
        self.temp_end = 0
        self.current_speech = {}
        self.finalized_segments = []
        self.current_frame_index = 0
        self.prev_end = 0
        self.next_start = 0

        # Counters for consecutive speech/no speech
        self.consecutive_speech = 0
        self.consecutive_no_speech = 0
        self.speech_chunks_without_process = 0
        self.n_frames_without_pause = 0

        self.pause_triggered = False  # Will turn True when a speech segment ends.

        # Reset model states
        self.vad_model.reset_states()

    def _ms_to_samples(self, ms: int) -> int:
        return int((ms / 1000) * self.sampling_rate)

    def process_frame(self, frame: torch.Tensor):
        """
        Process a single audio frame and update internal state.

        frame: torch.Tensor of shape [window_size_samples].
               If shorter, must be zero-padded before passing in.
        """
        # Compute speech probability

        # speech_prob = self.vad_model(frame, self.sampling_rate).item()
        speech_prob = self.vad_model.user_is_speaking_with_proba(frame)

        # Now apply logic similar to get_speech_timestamps
        i = self.current_frame_index
        frame_start_sample = i * self.window_size_samples
        self.current_frame_index += 1

        # Check maximum speech length
        if (
            self.triggered
            and (frame_start_sample - self.current_speech["start"])
            > self.max_speech_samples
        ):
            self._finalize_long_speech(frame_start_sample)
            return

        # If speech detected
        if speech_prob >= self.threshold:
            # If we had a temporary silence end, reset it
            if self.temp_end:
                self.temp_end = 0
                if self.next_start < self.prev_end:
                    self.next_start = frame_start_sample

            # Start speech if not already triggered
            if not self.triggered:
                self.triggered = True
                self.current_speech["start"] = frame_start_sample

        # If silence detected
        elif self.triggered and (speech_prob < self.neg_threshold):
            if not self.temp_end:
                self.temp_end = frame_start_sample

            # If silence is long enough
            if (frame_start_sample - self.temp_end) >= self.min_silence_samples:
                # Check if speech segment is long enough
                segment_len = self.temp_end - self.current_speech["start"]
                if segment_len > self.min_speech_samples:
                    # Finalize segment
                    self.current_speech["end"] = self.temp_end
                    self.finalized_segments.append(self.current_speech)
                    self._speech_segment_finalized()

                # Reset state
                self.current_speech = {}
                self.prev_end = 0
                self.next_start = 0
                self.temp_end = 0
                self.triggered = False
        if self.triggered:
            self.speech_chunks_without_process += 1
            self.consecutive_speech += 1
            self.consecutive_no_speech = 0
        else:
            self.consecutive_no_speech += 1
            self.consecutive_speech = 0
            self.n_frames_without_pause += 1

    def _finalize_long_speech(self, current_sample):
        """
        If speech runs longer than max_speech_duration, we end the segment aggressively.
        """
        if self.prev_end:
            self.current_speech["end"] = self.prev_end
            self.finalized_segments.append(self.current_speech)
            self.current_speech = {}
            if self.next_start < self.prev_end:
                self.triggered = False
            else:
                self.current_speech["start"] = self.next_start
        else:
            # Just end at current frame
            self.current_speech["end"] = current_sample
            self.finalized_segments.append(self.current_speech)
            self.current_speech = {}
            self.triggered = False

        self.prev_end = self.next_start = self.temp_end = 0
        self._speech_segment_finalized()

    def _speech_segment_finalized(self):
        """
        Mark that a speech segment was finalized, so should_pause() returns True.
        """
        self.pause_triggered = True

    def should_pause(self) -> bool:
        """
        Returns True if a speech segment was recently finalized.
        Call reset_pause_state() after handling the pause.
        """
        should_pause = self.pause_triggered
        if should_pause:
            self.reset_pause_state()
        return should_pause

    def reset_pause_state(self):
        """
        Reset pause trigger after handling a finalized segment.
        """
        self.pause_triggered = False
        self.finalized_segments.clear()

    def finalize(self):
        """
        Call at the end of the stream/audio. If still in a speech segment, finalize it if valid.
        """
        if self.triggered:
            end_sample = self.current_frame_index * self.window_size_samples
            if (
                end_sample - self.current_speech.get("start", 0)
            ) > self.min_speech_samples:
                self.current_speech["end"] = end_sample
                self.finalized_segments.append(self.current_speech)
                self._speech_segment_finalized()
            self.current_speech = {}
            self.triggered = False

    def get_segments(self):
        """
        Returns the finalized segments for inspection (start/end samples).
        """
        return self.finalized_segments

    def reset_pause_counters(self):
        self.consecutive_no_speech = 0
        self.n_frames_without_pause = 0
        self.speech_chunks_without_process = 0
        self.triggered = False

        self.consecutive_speech = 0
        self.consecutive_silence = 0
        self.in_speech = False

        # self.vad_model.reset_states()

    @property
    def is_speech(self):
        return self.triggered
        return self.triggered
