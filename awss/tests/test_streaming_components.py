import unittest
from unittest.mock import patch, MagicMock

import numpy as np

# Conditional imports for modules that might not be installed in the test environment
try:
    import librosa
except ImportError:
    librosa = MagicMock()

try:
    import torch
except ImportError:
    torch = MagicMock()

try:
    import whisper
except ImportError:
    whisper = MagicMock()

from awss.streaming.frames_chunk_policy import FramesChunkPolicy
from awss.streaming.webrtc_vad_model import WebRTCVAD
from awss.streaming.whisper_streaming import WhisperForStreaming


class TestFramesChunkPolicy(unittest.TestCase):
    def setUp(self):
        self.sample_audio_data = (np.random.rand(1600) * 32767).astype(np.int16).tobytes()

    @patch('librosa.core.resample')
    def test_resampling_skipped(self, mock_resample):
        policy = FramesChunkPolicy(source_sr=16000, models_sr=16000)
        policy.process_audio_frames(self.sample_audio_data)
        mock_resample.assert_not_called()
        self.assertEqual(len(policy.buffer_acumulated_for_asr), 1)
        self.assertIsInstance(policy.buffer_acumulated_for_asr[0], np.ndarray)
        self.assertEqual(policy.buffer_acumulated_for_asr[0].dtype, np.float32)

    @patch('librosa.core.resample')
    def test_resampling_performed(self, mock_resample):
        dummy_resampled_array = np.array([1, 2, 3], dtype=np.float32)
        mock_resample.return_value = dummy_resampled_array
        policy = FramesChunkPolicy(source_sr=48000, models_sr=16000)
        policy.process_audio_frames(self.sample_audio_data)
        mock_resample.assert_called_once()
        self.assertEqual(len(policy.buffer_acumulated_for_asr), 1)
        self.assertIs(policy.buffer_acumulated_for_asr[0], dummy_resampled_array)


class TestWebRTCVAD(unittest.TestCase):
    def setUp(self):
        self.dummy_byte_data = b'\x00\x01' * 160  # Sample 10ms frame for 16kHz

    @patch.object(WebRTCVAD, 'user_is_speaking')
    def test_speech_detected(self, mock_user_is_speaking):
        mock_user_is_speaking.return_value = True
        vad = WebRTCVAD(intensity=1, original_sr=16000)
        self.assertEqual(vad.user_is_speaking_with_proba(self.dummy_byte_data), 1.0)

    @patch.object(WebRTCVAD, 'user_is_speaking')
    def test_no_speech_detected(self, mock_user_is_speaking):
        mock_user_is_speaking.return_value = False
        vad = WebRTCVAD(intensity=1, original_sr=16000)
        self.assertEqual(vad.user_is_speaking_with_proba(self.dummy_byte_data), 0.0)

    @patch.object(WebRTCVAD, 'user_is_speaking')
    def test_exception_in_is_speech(self, mock_user_is_speaking):
        mock_user_is_speaking.side_effect = Exception("Test VAD exception")
        vad = WebRTCVAD(intensity=1, original_sr=16000)
        self.assertEqual(vad.user_is_speaking_with_proba(self.dummy_byte_data), 0.0)


class TestWhisperForStreaming(unittest.TestCase):
    def setUp(self):
        self.sample_audio_array = np.random.rand(16000).astype(np.float32)

    @patch('whisper.load_model')
    def test_model_name_loading(self, mock_load_model):
        WhisperForStreaming(model_name="test_model")
        mock_load_model.assert_called_once_with("test_model")

    @patch('whisper.load_model')
    def test_fp16_for_cuda(self, mock_load_model):
        mock_model_instance = MagicMock()
        mock_model_instance.transcribe = MagicMock(return_value={"text": "test"})
        mock_load_model.return_value = mock_model_instance

        # Mock torch.cuda.is_available to return True for this test
        with patch('torch.cuda.is_available', return_value=True):
            asr = WhisperForStreaming(model_name="any_model", device="cuda")
            asr.frames_to_text(self.sample_audio_array)
        
        # Check if device was set to cuda
        self.assertEqual(asr.device, "cuda")
        mock_model_instance.transcribe.assert_called_once_with(
            audio=unittest.mock.ANY, fp16=True
        )
        # Check that the first argument to transcribe (audio) is a numpy array
        args, _ = mock_model_instance.transcribe.call_args
        self.assertIsInstance(args[0], np.ndarray)


    @patch('whisper.load_model')
    def test_fp16_none_for_cpu(self, mock_load_model):
        mock_model_instance = MagicMock()
        mock_model_instance.transcribe = MagicMock(return_value={"text": "test"})
        mock_load_model.return_value = mock_model_instance
        
        # Mock torch.cuda.is_available to return False for this test
        with patch('torch.cuda.is_available', return_value=False):
            asr = WhisperForStreaming(model_name="any_model", device="cpu")
            asr.frames_to_text(self.sample_audio_array)

        # Check if device was set to cpu
        self.assertEqual(asr.device, "cpu")
        mock_model_instance.transcribe.assert_called_once_with(
            audio=unittest.mock.ANY, fp16=None
        )
        # Check that the first argument to transcribe (audio) is a numpy array
        args, _ = mock_model_instance.transcribe.call_args
        self.assertIsInstance(args[0], np.ndarray)


if __name__ == '__main__':
    unittest.main()
