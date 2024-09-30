from typing import Optional

import speech_recognition as sr


class SpeechRecognitionHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def recognize_from_file(
        self, audio_file_path: str, language: str = "en-US"
    ) -> Optional[str]:
        with sr.AudioFile(audio_file_path) as source:
            audio = self.recognizer.record(source)

        try:
            text = self.recognizer.recognize_whisper(audio, model="tiny")
            # text = self.recognizer.recognize_google(audio, language=language)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(
                f"Could not request results from Google Speech Recognition service; {e}"
            )
            return None

    def recognize_from_microphone(self, language: str = "en-US") -> Optional[str]:
        with sr.Microphone() as source:
            print("Say something!")
            audio = self.recognizer.listen(source)

        try:
            text = self.recognizer.recognize_google(audio, language=language)
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(
                f"Could not request results from Google Speech Recognition service; {e}"
            )
            return None


# Example usage
if __name__ == "__main__":
    handler = SpeechRecognitionHandler()

    # Recognize from file
    result = handler.recognize_from_file("/workspace/lex-levin-4min.wav")
    if result:
        print(f"Recognized text from file: {result}")

    # Recognize from microphone
    result = handler.recognize_from_microphone()
    if result:
        print(f"Recognized text from microphone: {result}")
