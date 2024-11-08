#!/usr/bin/python
import asyncio
import gc
import logging
import multiprocessing
import sys
from _thread import start_new_thread
from enum import Enum
from queue import Queue

import typer
import uvicorn
from fastapi import Depends, FastAPI, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from awss.meta.streaming_interfaces import ASRStreamingInterface
from awss.streaming.frames_chunk_policy import FramesChunkPolicy
from awss.streaming.silero_vad_model import SileroVAD
from awss.streaming.stream_manager import StreamManager
from awss.streaming.webrtc_vad_model import WebRTCVAD
from awss.streaming.whisper_streaming import WhisperForStreaming

# from awss.streaming.nemo_streaming import ConformerCTCForStreaming


def load_custom_model(model_name) -> ASRStreamingInterface:
    import os
    from importlib import import_module

    from dotenv import load_dotenv

    load_dotenv()

    custom_model_path = os.getenv("CUSTOM_MODEL_PATH")

    if custom_model_path:
        module_path, class_name = custom_model_path.rsplit(".", 1)
        module = import_module(module_path)
        CustomModelClass = getattr(module, class_name)

        assert issubclass(
            CustomModelClass, ASRStreamingInterface
        ), "CustomModelClass must implement ASRStreamingInterface"
        return CustomModelClass(model_name=model_name)

    else:
        raise ValueError("CUSTOM_MODEL_PATH not set in .env file")


class UvicornServer(multiprocessing.Process):
    def __init__(self, config: uvicorn.Config):
        super().__init__()

        self.config = config

    def stop(self):
        self.terminate()

    def run(self, *args, **kwargs):
        server = uvicorn.Server(config=self.config)
        server.run()


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

POLICIES = {
    "buffered": FramesChunkPolicy,
}
Policy = Enum(value="Policy", names=[(k, k) for k in POLICIES])


VAD_LOADER = {"webrtcvad": WebRTCVAD, "silerovad": SileroVAD}
VADModel = Enum(value="VADModel", names=[(k, k) for k in VAD_LOADER])


# MODELS = {"nemo": ConformerCTCForStreaming, "whisper": WhisperForStreaming}
MODELS = {"whisper": WhisperForStreaming, "custom": load_custom_model}
Model = Enum(value="Model", names=[(k, k) for k in MODELS])

STREAM_MANAGERS = {"default": StreamManager}

STREAM_MANAGER = Enum(value="StreamManager", names=[(k, k) for k in STREAM_MANAGERS])


app = typer.Typer()


def read_outqueue(
    stream_manager: StreamManager,
    out_queue: Queue,
    websocket: WebSocket,
):
    while websocket.state != WebSocketState.DISCONNECTED:
        response = stream_manager.get_last_text()
        if response == "close":
            break
        out_queue.put_nowait(response)


def dummy_manager_init() -> StreamManager:
    raise NotImplementedError("This function must be overwrited")


async def websocket_endpoint(
    websocket: WebSocket,
    source_sr: int = Query(
        None,
        description="source sample rate, default None (it use the default server parameter)",
    ),
    vad_sr: int = Query(
        None,
        description="VAD sample rate, default None (it use the default server parameter)",
    ),
    stream_manager=Depends(dummy_manager_init),
):

    await websocket.accept()

    in_queue = Queue()
    print(f"source_sr => {source_sr}, vad_sr => {vad_sr}")
    stream_manager.update_params(source_sr=source_sr, vad_sr=vad_sr)
    stream_manager.start(lambda: in_queue.get())

    out_queue = asyncio.Queue()
    out_queue.join()
    start_new_thread(read_outqueue, (stream_manager, out_queue, websocket))
    try:
        while websocket.state != WebSocketState.DISCONNECTED:

            data = await websocket.receive_bytes()
            in_queue.put(data)
            if not out_queue.empty():
                json_ = out_queue.get_nowait()
                logger.info(f"response: {json_}")
                await websocket.send_json(json_)

    except WebSocketDisconnect:
        print("server >> client left chat.")
    finally:
        in_queue.put("close")
        stream_manager.stop()
        gc.collect()


@app.command()
def cli(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        metavar="HOST",
        help="host",
    ),
    port: int = typer.Option(
        23000,
        "--port",
        "-p",
        metavar="PORT",
        help="port",
    ),
    model: Model = typer.Option(
        "whisper",
        help="ASR pipeline to use",
    ),
    model_name: str = typer.Option(
        "tiny",
        help="ASR pipeline to use",
    ),
    policy: Policy = typer.Option(
        "buffered",
        help="Stream Policy to use",
        case_sensitive=False,
    ),
    vad: VADModel = typer.Option(
        "silerovad",
        help="VAD model to use",
    ),
    strm_mgr: STREAM_MANAGER = typer.Option(
        "default",
        help="Stream manager to use",
    ),
    source_sr: int = typer.Option(
        16_000,
        help="SampleRate sended by client",
    ),
    asr_sr: int = typer.Option(
        16_000,
        help="SampleRate of the ASR model",
    ),
    vad_sr: int = typer.Option(
        16_000,
        help="SampleRate of the VAD model",
    ),
    partial_results: bool = typer.Option(
        False,
        help="Enable partial results",
    ),
    exclude_silence_chunks: bool = typer.Option(
        False,
        help="Enable filter chunks with silences",
    ),
):
    vad = vad.name
    policy = policy.name
    logger.info(model.name)

    model = MODELS[model.name](model_name)

    # model = ConformerCTCForStreaming(model_name)

    def stream_manager_init():
        logger.info(f"Initializing StreamManager with {policy} policy")
        # vad_model = VAD_LOADER[vad](2, vad_sr)
        vad_model = SileroVAD(0, vad_sr)
        chunk_policy = POLICIES[policy](
            source_sr, asr_sr, 4 if partial_results else 1e10
        )
        # manager = STREAM_MANAGERS[strm_mgr](model, vad_model, chunk_policy, source_sr)
        # return manager
        return StreamManager(
            model,
            vad_model,
            chunk_policy,
            source_sr,
            exclude_silences=exclude_silence_chunks,
        )
        # return SpeechRecognitionStreamManager(model, vad_model, chunk_policy, source_sr)

    logger.info("Starting stream server...")
    server = FastAPI()

    server.websocket("/ws")(websocket_endpoint)

    server.dependency_overrides[dummy_manager_init] = stream_manager_init
    config = uvicorn.Config(
        server,
        host=host,
        port=port,
        log_level="info",
        loop="asyncio",
    )
    server = UvicornServer(config=config)
    server.run()


if __name__ == "__main__":
    app()
