import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.title("🐶 Dog Bark Emotion Classifier (Prototype)")

# File uploader
uploaded_file = st.file_uploader("Upload a dog bark (.wav) file", type=["wav"])

if uploaded_file:
    # Show audio player
    st.audio(uploaded_file)

    # Dummy prediction
    st.info("This is a test prototype – no prediction logic yet.")
    st.success("🐶 Predicted emotion: **(placeholder)**")
else:
    st.info("Please upload a .wav file to see the demo.")

webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": False, "audio": True}
)

