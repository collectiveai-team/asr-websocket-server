#!/usr/bin/python
import io
import time
import logging
from pathlib import Path
from _thread import start_new_thread

import typer
import librosa
import soundfile as sf
from fastapi import WebSocketDisconnect
from websocket import create_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def cli(
    audio_path: Path = typer.Argument(
        ...,
        help="path to file to stream",
    ),
    endpoint: str = typer.Option(
        "ws://0.0.0.0:23000/ws",
        help="server endpoint",
    ),
    source_sr: int = typer.Option(
        16_000,
        help="SampleRate sended by client",
    ),
    frame_duration: int = typer.Option(
        20,
        help="frame duration in ms to send to server",
    ),
    params: list[str] = typer.Option(
        [],
        help="Additional parameters for the prediction",
    ),
):
    # Setting buffer length
    buffer_length = int(source_sr * frame_duration / 1000)
    client = create_connection(f'{endpoint}?{"&".join(params)}')

    responses = []

    def read_transcript(client):
        while True:
            try:
                response = client.recv()
                logger.info(f"reciving response:\n\t{response}\n")
                responses.append(response)
            except WebSocketDisconnect:
                pass
            except Exception:
                pass

    # Opening the audio file
    signal, sr = librosa.load(audio_path, sr=source_sr, mono=True)
    duration = librosa.get_duration(signal, sr=source_sr)
    buffer = io.BytesIO()
    sf.write(buffer, signal, samplerate=sr, subtype="PCM_16", format="wav")
    buffer.seek(0)

    start_new_thread(read_transcript, (client,))
    # While loop for the transfer of file
    start_time = time.time()
    while data := buffer.read(buffer_length):
        if client.send_binary(data):
            # We divide by 2 to correlate the waiting time with the audio duration
            time.sleep(frame_duration / 1000 / 2)  # waiting for 20 miliseconds
    logger.info(
        f"It tooks {time.time() - start_time} to reproduce the audio of {duration}"
    )
    client.close()


if __name__ == "__main__":
    app()
