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


def setup_connection(endpoint, params, source_sr, frame_duration):
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
                break
            except Exception:
                continue

    start_new_thread(read_transcript, (client,))
    return client, buffer_length, responses


def stream_audio(client, audio_path, source_sr, frame_duration, buffer_length):
    signal, sr = librosa.load(audio_path, sr=source_sr, mono=True)
    duration = librosa.get_duration(signal, sr=source_sr)
    buffer = io.BytesIO()
    sf.write(buffer, signal, samplerate=sr, subtype="PCM_16", format="wav")
    buffer.seek(0)
    logger.info(
        f"streaming audio of {duration} seconds, with buffer length {buffer_length} and frame duration {frame_duration} and source sr {source_sr}"
    )
    start_time = time.time()
    while data := buffer.read(buffer_length):
        if client.send_binary(data):
            time.sleep(frame_duration / 1000 / 2)
    logger.info(
        f"It tooks {time.time() - start_time} to reproduce the audio of {duration}"
    )
    client.close()


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
        32,
        help="frame duration in ms to send to server",
    ),
    params: list[str] = typer.Option(
        [],
        help="Additional parameters for the prediction",
    ),
):
    client, buffer_length, responses = setup_connection(
        endpoint, params, source_sr, frame_duration
    )
    stream_audio(client, audio_path, source_sr, frame_duration, buffer_length)

    if responses:
        logger.info(f"Transcript: \n{responses[-1]}")


if __name__ == "__main__":
    app()
