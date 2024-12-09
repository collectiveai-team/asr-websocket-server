import os
from collections import deque

from awss.logger import get_logger
from awss.meta.streaming_interfaces import VADModelInterface

logger = get_logger(__name__)


class EnhancedSpeechFrameDetector:
    def __init__(
        self,
        vad_model: VADModelInterface,
        base_threshold: float = 0.5,
        sampling_rate: int = 16000,
        frame_duration_ms: int = 10,
        # Strategy toggles
        use_hangover: bool = True,
        speech_start_hangfront: int = 3,
        speech_end_hangover: int = 5,
        use_sliding_window: bool = False,
        sliding_window_size: int = 5,
        use_onset_offset_thresholds: bool = False,
        speech_onset_threshold: float = 0.6,
        speech_offset_threshold: float = 0.4,
        use_time_based_silence: bool = False,
        min_silence_ms: int = 300,
        use_adaptive_thresholds: bool = False,
        adaptive_increase_step: float = 0.1,
        adaptive_decrease_step: float = 0.05,
        max_threshold: float = 0.9,
        min_threshold: float = 0.3,
        # Original counters and thresholds
        n_frames_without_pause_min_threshold: int = None,
        n_frames_without_pause_max_threshold: int = None,
        consecutive_no_speech_lower_threshold: int = None,
        consecutive_no_speech_upper_threshold: int = None,
    ):
        self.vad_model = vad_model
        self.sampling_rate = sampling_rate
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold

        # Strategies
        self.use_hangover = use_hangover
        self.speech_start_hangfront = speech_start_hangfront
        self.speech_end_hangover = speech_end_hangover

        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size
        self.prob_window = (
            deque(maxlen=sliding_window_size) if use_sliding_window else None
        )

        self.use_onset_offset_thresholds = use_onset_offset_thresholds
        self.speech_onset_threshold = speech_onset_threshold
        self.speech_offset_threshold = speech_offset_threshold

        self.use_time_based_silence = use_time_based_silence
        self.frame_duration_ms = frame_duration_ms
        self.min_silence_ms = min_silence_ms
        self.silence_frames_needed = (
            min_silence_ms // frame_duration_ms if use_time_based_silence else None
        )

        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.adaptive_increase_step = adaptive_increase_step
        self.adaptive_decrease_step = adaptive_decrease_step
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold

        # State tracking
        self.consecutive_speech = 0
        self.consecutive_silence = 0
        self.in_speech = False

        self.n_frames_without_pause = 0
        self.speech_chunks_without_process = 0
        self.triggered = False
        self.consecutive_no_speech = 0

        # Environment variables for thresholds
        self.n_frames_without_pause_min_threshold = (
            n_frames_without_pause_min_threshold
            or int(os.getenv("N_FRAMES_WITHOUT_MIN_PAUSE_THRESHOLD", "150"))
        )
        self.n_frames_without_pause_max_threshold = (
            n_frames_without_pause_max_threshold
            or int(os.getenv("N_FRAMES_WITHOUT_MAX_PAUSE_THRESHOLD", "450"))
        )
        self.consecutive_no_speech_lower_threshold = (
            consecutive_no_speech_lower_threshold
            or int(os.getenv("CONSECUTIVE_NO_SPEECH_LOWER_THRESHOLD", "1"))
        )
        self.consecutive_no_speech_upper_threshold = (
            consecutive_no_speech_upper_threshold
            or int(os.getenv("CONSECUTIVE_NO_SPEECH_UPPER_THRESHOLD", "70"))
        )

    def process_frame(self, frame):
        speech_prob = self.vad_model.user_is_speaking_with_proba(frame)

        # Apply sliding window smoothing if enabled
        if self.use_sliding_window:
            self.prob_window.append(speech_prob)
            avg_prob = sum(self.prob_window) / len(self.prob_window)
        else:
            avg_prob = speech_prob

        # Determine preliminary speech decision based on current thresholds
        frame_is_speech = self._apply_thresholds(avg_prob)

        # Update counters (consecutive_speech, consecutive_silence) based on this frame
        self._update_counters_for_frame(frame_is_speech)

        # Apply hangover/time-based silence logic to finalize in_speech decision
        self._apply_hangover_logic()

        # Apply optional adaptive thresholding
        self._apply_adaptive_thresholds()

        # Update original logic counters
        self._update_original_counters()

        if self.n_frames_without_pause % 100 == 0:
            logger.info(f"Speech Probability: {speech_prob}")
            logger.info(f"Average Probability: {avg_prob}")
            logger.info(f"Current Threshold: {self.current_threshold}")
            logger.info(f"Consecutive Speech: {self.consecutive_speech}")
            logger.info(f"Consecutive Silence: {self.consecutive_silence}")
            logger.info(f"In Speech: {self.in_speech}")
            logger.info(
                f"Speech Chunks Without Process: {self.speech_chunks_without_process}"
            )
            logger.info(f"Triggered: {self.triggered}")

    def _apply_thresholds(self, avg_prob: float) -> bool:
        # If using onset/offset thresholds
        if self.use_onset_offset_thresholds:
            if self.in_speech:
                # Use offset threshold
                return avg_prob >= self.speech_offset_threshold
            else:
                # Use onset threshold
                return avg_prob >= self.speech_onset_threshold
        else:
            # Use a single threshold (current_threshold)
            if self.in_speech:
                # Already in speech, continue speech if above current threshold
                return avg_prob >= self.current_threshold
            else:
                # Out of speech, start speech if above current threshold
                return avg_prob >= self.current_threshold

    def _update_counters_for_frame(self, frame_is_speech: bool):
        if frame_is_speech:
            self.consecutive_speech += 1
            self.consecutive_silence = 0
        else:
            self.consecutive_silence += 1
            self.consecutive_speech = 0

    def _apply_hangover_logic(self):
        # If using hangover/time-based silence, determine in_speech based on counters
        if self.use_hangover:
            if (
                not self.in_speech
                and self.consecutive_speech >= self.speech_start_hangfront
            ):
                # Enough consecutive speech to start speaking
                self.in_speech = True

            if self.in_speech:
                # To stop speech, either use time-based silence or hangover
                if self.use_time_based_silence:
                    if self.consecutive_silence >= self.silence_frames_needed:
                        self.in_speech = False
                else:
                    if self.consecutive_silence >= self.speech_end_hangover:
                        self.in_speech = False
        else:
            # Without hangover, if time-based silence is enabled:
            if (
                self.use_time_based_silence
                and self.in_speech
                and self.consecutive_silence >= self.silence_frames_needed
            ):
                self.in_speech = False
            # If no hangover and no time-based silence, in_speech was directly determined by thresholds.
            # But since we first updated counters, if frame_is_speech was False, we might have left speech.
            # This case would need no extra logic because _apply_thresholds + counters handle it.

    def _apply_adaptive_thresholds(self):
        if self.use_adaptive_thresholds:
            # Example heuristic:
            # If in speech, gradually lower threshold to avoid losing speech easily
            # If in silence, gradually raise threshold to avoid false triggers
            # if self.in_speech:
            #     self.current_threshold = max(
            #         self.min_threshold,
            #         self.current_threshold - self.adaptive_decrease_step,
            #     )
            # else:
            #     self.current_threshold = min(
            #         self.max_threshold,
            #         self.current_threshold + self.adaptive_increase_step,
            #     )
            if self.in_speech:
                self.current_threshold = min(
                    self.max_threshold,
                    self.current_threshold + self.adaptive_increase_step,
                )
            else:
                self.current_threshold = max(
                    self.min_threshold,
                    self.current_threshold - self.adaptive_decrease_step,
                )

    def _update_original_counters(self):
        self.n_frames_without_pause += 1

        if self.in_speech:
            self.consecutive_no_speech = 0
            self.speech_chunks_without_process += 1
            if not self.triggered:
                self.triggered = True
        else:
            self.consecutive_no_speech += 1
            # Reset states if too much silence, as in original code
            if (
                self.consecutive_no_speech
                > 10 * self.consecutive_no_speech_upper_threshold
            ):
                self.vad_model.reset_states()

    def should_pause(self):
        # If we're using hangover/time-based silence logic, check if we've stabilized silence
        if not self.in_speech:
            # Time-based silence check:
            if (
                self.use_time_based_silence
                and self.consecutive_silence >= self.silence_frames_needed
                and self.speech_chunks_without_process > 0
            ):
                return True

            # Hangover logic (no time-based silence):
            if self.use_hangover and not self.use_time_based_silence:
                if (
                    self.consecutive_silence >= self.speech_end_hangover
                    and self.speech_chunks_without_process > 0
                ):
                    return True

        # As a secondary condition, if we've spoken a lot (speech_chunks_without_process > 1)
        # and then have a long silence:
        if (
            self.consecutive_no_speech > self.consecutive_no_speech_upper_threshold
            and self.speech_chunks_without_process > 1
        ):
            return True

        # Hard limit fallback:
        if self.n_frames_without_pause > self.n_frames_without_pause_max_threshold:
            return True

        # Pause Trigger on Transition Events:
        if (
            not self.in_speech
            and self.consecutive_silence >= self.silence_frames_needed
            and self.speech_chunks_without_process > 0
        ):
            return True

        return False

    def reset_pause_counters(self):
        self.consecutive_no_speech = 0
        self.n_frames_without_pause = 0
        self.speech_chunks_without_process = 0
        self.current_threshold = self.base_threshold
        self.triggered = False

        self.consecutive_speech = 0
        self.consecutive_silence = 0
        self.in_speech = False
        if self.use_sliding_window:
            self.prob_window.clear()

        self.vad_model.reset_states()

    def is_currently_in_speech(self):
        return self.in_speech

    @property
    def is_speech(self):
        return self.in_speech
