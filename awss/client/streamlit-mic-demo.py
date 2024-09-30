import os
import time
import queue
import logging
import threading
import logging.handlers

import pydub
import websocket
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from fastapi import WebSocketDisconnect
from websocket import create_connection
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from streamlit.runtime.scriptrunner.script_run_context import add_script_run_ctx

logger = logging.getLogger(__name__)


def main():
    st.header("Real Time Speech-to-Text")
    config = st.expander("Configuration", expanded=False)
    websocket_endpoint = os.environ.get("WEBSOCKET_ENDPOINT", "ws://fastapi:23000/ws")
    with config:
        endpoint = st.text_input("Websocket endpoint", websocket_endpoint)
        sample_rate = st.number_input(
            "Websocket sample rate",
            min_value=8000,
            max_value=48000,
            value=16000,
            step=8000,
        )
        chunk_size_ms = st.number_input(
            "Buffer length [ms]",
            min_value=10,
            max_value=30,
            value=20,
            step=10,
        )

    BUFFER_SIZE = chunk_size_ms * sample_rate / 1000
    endpoint = f"{endpoint}?&source_sr={sample_rate}&vad_sr={sample_rate}"

    logger.info(f"websocket endpoint: {endpoint}")
    logger.info(f"websocket sample rate: {sample_rate}")
    logger.info(f"websocket buffer length: {chunk_size_ms}")
    logger.info(f"websocket buffer size: {BUFFER_SIZE}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 2))

    webrtc_ctx = webrtc_streamer(
        key="streamer",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=int(BUFFER_SIZE),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )
    status_indicator = st.empty()
    plot = st.empty()
    output = st.empty()
    output.markdown("**Response:**")
    history = st.expander("History")
    with history:
        st.button("clear", on_click=lambda *x: output.empty())

    ax.cla()
    ax.plot([0])
    ax.set_ylim(ymin=-33000, ymax=33000)
    ax.set_axis_off()
    with plot:
        st.pyplot(fig, transparent=True)

    status_indicator.warning("Loading...")

    if not webrtc_ctx.state.playing:
        return

    # connection to websocket
    # URI = f'{ENDPOINT}?{"&".join(PARAMS)}'
    websocket.enableTrace(False)
    ws = create_connection(endpoint)

    def read_transcript(client):
        while True:
            try:
                response = client.recv()
                logger.info(f"reciving response:\n\t{response}\n")
                output.markdown(f"**Response:** {response}")
                history.markdown(f"**Response:** {response}")
            except WebSocketDisconnect:
                pass
            except Exception as e:
                output.error(e)

    thread = threading.Thread(target=read_transcript, args=(ws,))
    add_script_run_ctx(thread)
    thread.start()

    while True:
        status_indicator.warning("Running. Say something!")
        chunk = pydub.AudioSegment.empty()
        try:
            stream = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            time.sleep(0.1)
            status_indicator.warning("No frame arrived.")
            continue

        for audio_frame in stream:
            sound = pydub.AudioSegment(
                data=audio_frame.to_ndarray().tobytes(),
                sample_width=audio_frame.format.bytes,
                frame_rate=audio_frame.sample_rate,
                channels=len(audio_frame.layout.channels),
            )
            chunk += sound

        if len(chunk) > 0:
            chunk = chunk.set_channels(1).set_frame_rate(int(sample_rate))
            buffer = np.array(chunk.get_array_of_samples())
            for frame in np.array_split(buffer, len(stream)):
                ws.send(frame.tobytes(), websocket.ABNF.OPCODE_BINARY)

            # audio display
            ax.cla()
            ax.plot(chunk.get_array_of_samples())
            ax.set_ylim(ymin=-33000, ymax=33000)
            ax.set_axis_off()
            with plot:
                st.pyplot(fig, transparent=True)


if __name__ == "__main__":
    main()
