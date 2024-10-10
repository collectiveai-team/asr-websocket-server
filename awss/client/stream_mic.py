# client.py

import time
import threading

import socketio
import speech_recognition as sr

# Configuration
SERVER_URL = "http://localhost:5000"  # Change if server is hosted elsewhere
SAMPLE_RATE = 16000  # Must match server's sample rate
CHUNK_TIMEOUT = 2  # Duration in seconds for each audio chunk

# Initialize SocketIO client
sio = socketio.Client()


def on_transcription(data):
    text = data.get("text", "")
    print(f"Transcription: {text}")


def on_response(data):
    message = data.get("message", "")
    print(f"Server: {message}")


def on_connect():
    print("Connected to the server")


def on_disconnect():
    print("Disconnected from the server")


# Register event handlers
sio.on("transcription", on_transcription)
sio.on("response", on_response)
sio.on("connect", on_connect)
sio.on("disconnect", on_disconnect)


def record_callback(recognizer, audio):
    """
    Callback function called from the background thread when audio is captured.
    """
    try:
        # Get raw audio data
        audio_bytes = audio.get_raw_data()
        print(f"Captured audio chunk of {len(audio_bytes)} bytes. Sending to server...")

        # Send audio bytes to the server
        sio.emit("audio", {"audio": audio_bytes})

    except Exception as e:
        print(f"Error in record_callback: {e}")


def start_listening():
    """
    Starts the background listening thread using SpeechRecognition's listen_in_background.
    """
    recognizer = sr.Recognizer()
    mic = sr.Microphone(sample_rate=SAMPLE_RATE)

    with mic as source:
        print("Calibrating microphone for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Calibration complete. Start speaking.")

    # Start background listening
    stop_listening = recognizer.listen_in_background(
        mic, record_callback, phrase_time_limit=CHUNK_TIMEOUT
    )
    print("Started background listening.")

    try:
        while True:
            time.sleep(0.1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Stopping background listening...")
        stop_listening(wait_for_stop=False)


if __name__ == "__main__":
    try:
        # Connect to the server
        sio.connect(SERVER_URL)
    except Exception as e:
        print(f"Unable to connect to the server: {e}")
        exit(1)

    # Start background listening in a separate thread
    listening_thread = threading.Thread(target=start_listening)
    listening_thread.start()

    # Keep the main thread alive to listen for events
    try:
        while listening_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user. Disconnecting...")
    finally:
        sio.disconnect()
        print("Disconnected from the server.")
        print("Disconnected from the server.")
