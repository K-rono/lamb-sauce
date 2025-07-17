import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import joblib
import numpy as np
import librosa
import tempfile
import av
import queue
import threading
import time

st.title("🐶 Dog Bark Emotion Classifier (Prototype)")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("dog_emotion_model.pkl")  # Load your model

model = load_model()

emotion_labels = ['Aggressive', 'Alert', 'Excited', 'Frustrated', 'Happy', 'Neutral', 'Playful', 'Sigh', 'Whining']

audio_buffer = queue.Queue()
current_prediction = {"label": "No prediction yet"}

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        #Extract audio numpy array
        audio_data = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        audio_buffer.put(audio_data)
        return frame

def extract_realtime_features(audio_np):
    #Resample from 48kHz to 22.05kHz
    audio_resampled = librosa.resample(audio_np, orig_sr=48000, target_sr=22050)
    
    #Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio_resampled, sr=22050, n_mfcc=100)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
    return mfcc_mean

def predict_loop():
    while True:
        time.sleep(2)  # Predict every 5 seconds
        audio_chunks = []
        while not audio_buffer.empty():
            audio_chunks.append(audio_buffer.get())
        if len(audio_chunks) == 0:
            continue
        audio_np = np.concatenate(audio_chunks)
        try:
            features = extract_realtime_features(audio_np)
            prediction = model.predict(features)[0]
            predicted_label = emotion_labels[int(prediction)]
            current_prediction["label"] = f"🐾 Detected Emotion: {predicted_label}"
        except Exception as e:
            current_prediction["label"] = f"Error: {e}"


webrtc_ctx = webrtc_streamer(
    key="dog-audio",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": False, "audio": True}
)

prediction_display = st.empty()

if webrtc_ctx.state.playing:
    st.info("🎤 Listening and predicting every 5 seconds...")
    threading.Thread(target=predict_loop, daemon=True).start()
    while True:
        prediction_display.info(current_prediction["label"])
        time.sleep(1)  # Refresh UI every second
else:
    st.warning("Please start recording to begin prediction.")





# file uploaded method start

# def extract_features(file_path):
#     y, sr = librosa.load(file_path, sr=22050)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)
#     mfcc_mean = np.mean(mfcc.T, axis=0)
#     return mfcc_mean.reshape(1, -1)  # Will return shape (1, 100)

# # File uploader
# uploaded_file = st.file_uploader("Upload a dog bark (.wav) file", type=["wav"])

# if uploaded_file:
#     st.audio(uploaded_file)

#     # Save the uploaded file to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         tmp_path = tmp_file.name

#     # Extract features
#     features = extract_features(tmp_path)

#     # Scale if needed
#     # features = scaler.transform(features)  # Uncomment if using scaler

#     # Predict
#     predicted_index = model.predict(features)[0]
#     predicted_label = emotion_labels[int(predicted_index)]

#     # Display result
#     st.success(f"🐶 Predicted emotion: **{predicted_label}**")
# else:
#     st.info("Please upload a .wav file to get prediction.")

# file uploaded method end




# if uploaded_file:
#     # Show audio player
#     st.audio(uploaded_file)

#     # Dummy prediction
#     st.info("This is a test prototype – no prediction logic yet.")
#     st.success("🐶 Predicted emotion: **(placeholder)**")
# else:
#     st.info("Please upload a .wav file to see the demo.")

