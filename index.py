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
import audio_state
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import soundfile as sf


st.title("🐶 Dog Bark Emotion Classifier (Prototype)")

# # Load random forest model
# @st.cache_resource
# def load_model():
#     return joblib.load("dog_emotion_model.pkl")  # Load your model

# model = load_model()

# Load LSTM model
@st.cache_resource
def load_lstm_model():
    return keras.models.load_model("final_emotion_model.keras") 

model = load_lstm_model()

@st.cache_resource
def load_arousal_model():
    return keras.models.load_model("final_arousal_model.keras")

arousal_model = load_arousal_model()

@st.cache_resource
def load_valence_model():
    return keras.models.load_model("final_valence_model.keras")

valence_model = load_valence_model()


#emotion_labels = ['Aggressive', 'Alert', 'Excited', 'Frustrated', 'Happy', 'Neutral', 'Playful', 'Sigh', 'Whining']


# class AudioProcessor(AudioProcessorBase):
#     def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
#         audio_data = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
#         audio_state.audio_buffer.put(audio_data)
#         #print("Audio received:", audio_data.shape)
#         return frame

# def extract_realtime_features(audio_np):
#     #Resample from 48kHz to 22.05kHz
#     audio_resampled = librosa.resample(audio_np, orig_sr=48000, target_sr=22050)
    
#     #Extract MFCC
#     mfcc = librosa.feature.mfcc(y=audio_resampled, sr=22050, n_mfcc=100)
#     mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
#     return mfcc_mean

# def predict_loop():
#     while True:
#         time.sleep(2)  # Predict every 5 seconds
#         audio_chunks = []

#         while not audio_state.audio_buffer.empty():
#             audio_chunks.append(audio_state.audio_buffer.get())
            
#         if len(audio_chunks) == 0:
#             continue

#         audio_np = np.concatenate(audio_chunks)
#         try:
#             features = extract_realtime_features(audio_np)
#             prediction = model.predict(features)[0]
#             predicted_label = emotion_labels[int(prediction)]
#             audio_state.current_prediction["label"] = f"🐾 Detected Emotion: {predicted_label}"
#         except Exception as e:
#             audio_state.current_prediction["label"] = f"Error: {e}"

# if "prediction_thread_started" not in st.session_state:
#     threading.Thread(target=predict_loop, daemon=True).start()
#     st.session_state.prediction_thread_started = True

# webrtc_ctx = webrtc_streamer(
#     key="dog-audio",
#     mode=WebRtcMode.SENDRECV,
#     audio_processor_factory=AudioProcessor,
#     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#     media_stream_constraints={"video": False, "audio": True}
# )

# # if not audio_buffer.empty():
# #     st.success(f"Buffer has {audio_buffer.qsize()} chunks")
# # else:
# #     st.warning("Buffer is empty. Speak into your mic to test.")

# # prediction_display = st.empty()

# # if webrtc_ctx.state.playing:
# #     st.info("🎤 Listening and predicting every 5 seconds...")
# #     threading.Thread(target=predict_loop, daemon=True).start()
# #     while True:
# #         prediction_display.info(current_prediction["label"])
# #         time.sleep(1)  # Refresh UI every second
# # else:
# #     st.warning("Please start recording to begin prediction.")

# # if webrtc_ctx.state.playing:
# #     st.success("✅ Microphone is streaming.")

# #     buf = audio_state.audio_buffer

# #     if not buf.empty():
# #         st.success(f"✅ Buffer has {buf.qsize()} chunks.")
# #         audio_chunk = buf.get()
# #         st.write("First 10 values:", audio_chunk[:10])
# #         st.line_chart(audio_chunk[:500])
# #     else:
# #         st.warning("🔇 Buffer is still empty. Try speaking into the mic.")
# # else:
# #     st.info("Click Start above to begin recording.")

# if webrtc_ctx.state.playing:
#     st.success("🎤 Microphone is streaming. Listening...")
# else:
#     st.info("Click Start to begin recording.")

# # Show prediction result
# st.subheader("Prediction Result:")
# st.info(audio_state.current_prediction["label"])    

# file uploaded method start

# def extract_features(file_path): OLDDDDDDDDDDDDDDDDD
#     y, sr = librosa.load(file_path, sr=22050)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)
#     mfcc_mean = np.mean(mfcc.T, axis=0)
#     return mfcc_mean.reshape(1, -1)  # Will return shape (1, 100)


    # y, sr = librosa.load(file_path, sr=22050)
    
    # # Get MFCC with 100 coefficients (shape: [n_mfcc, time_steps])
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)

    # # Transpose to shape: (time_steps, n_mfcc)
    # mfcc = mfcc.T  # shape: (time_steps, 100)

    # # Optional: fix the time_steps dimension (e.g., pad or truncate to 128)
    # desired_time_steps = 128
    # if mfcc.shape[0] < desired_time_steps:
    #     # Pad with zeros
    #     pad_width = desired_time_steps - mfcc.shape[0]
    #     mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    # elif mfcc.shape[0] > desired_time_steps:
    #     # Truncate
    #     mfcc = mfcc[:desired_time_steps, :]

    # # Final shape: (1, time_steps, n_mfcc)
    # return mfcc.reshape(1, desired_time_steps, 100)

# Define the parameters used for feature extraction and processing in your training
# Ensure these match the parameters used in cell 13a82ffa where X_sequential_scaled was created
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 1 # Crucial: Match the n_mfcc used for the sequential models
SEQUENCE_LENGTH = 25 # Crucial: Match the sequence length used for the sequential models

# Assuming you have a fitted scaler for sequential data ('scaler_sequential')
# You will need to load or fit this scaler on your training data in your Streamlit app
# For demonstration purposes here, we'll assume it's available or loaded.
# In your Streamlit app, you'd typically load the fitted scaler like:
# scaler_sequential = joblib.load('scaler_sequential.pkl') # Example using joblib

def preprocess_audio_for_sequential_model(audio_file_path, scaler):
    """
    Preprocesses an audio file for input to the sequential (LSTM) Arousal/Valence models.

    Args:
        audio_file_path (str): The full path to the audio file.
        scaler (sklearn.preprocessing.StandardScaler): The scaler fitted on the sequential training data.

    Returns:
        np.ndarray: The preprocessed and scaled feature sequence, ready for model input
                    (shape: (1, SEQUENCE_LENGTH, N_MFCC)), or None if processing fails.
    """
    try:
        # Load audio
        y, sr_loaded = librosa.load(audio_file_path, sr=SR) # Ensure consistent sample rate

        # Extract MFCCs over time (using n_mfcc=1)
        mfccs = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                                    n_fft=N_FFT, hop_length=HOP_LENGTH)

        # Transpose MFCCs to be (timesteps, features) -> (timesteps, 1) since n_mfcc is 1
        mfccs = mfccs.T # Shape will be (timesteps, 1)

        # Handle sequence length: Pad or truncate to SEQUENCE_LENGTH
        if mfccs.shape[0] < SEQUENCE_LENGTH:
            # Pad with a constant value (e.g., 0 or the mean of the data)
            # Using 0 padding for simplicity here, must match training padding strategy
            padding_needed = SEQUENCE_LENGTH - mfccs.shape[0]
            # Pad along the time axis (axis 0)
            mfccs_padded = np.pad(mfccs, ((0, padding_needed), (0, 0)), mode='constant')
            processed_sequence = mfccs_padded
        else:
            # Truncate the sequence
            processed_sequence = mfccs[:SEQUENCE_LENGTH, :]

        # Ensure the processed sequence has the shape (SEQUENCE_LENGTH, N_MFCC) -> (25, 1)
        if processed_sequence.shape != (SEQUENCE_LENGTH, N_MFCC):
            st.info(f"Warning: Processed sequence has unexpected shape {processed_sequence.shape}. Expected {(SEQUENCE_LENGTH, N_MFCC)}")
            return None # Cannot predict with wrong shape

        # Scale the features using the *fitted* scaler
        # The scaler expects input shape (-1, number_of_features). Since N_MFCC is 1,
        # we can pass the (SEQUENCE_LENGTH, 1) shaped array directly to transform.
        processed_sequence_scaled = scaler.transform(processed_sequence)

        # Add a batch dimension for model prediction: (timesteps, features) -> (1, timesteps, features)
        input_data = np.expand_dims(processed_sequence_scaled, axis=0)

        # Convert to TensorFlow tensor (optional, but good practice for TF models)
        input_data_tf = tf.convert_to_tensor(input_data, dtype=tf.float32)

        return input_data_tf

    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_file_path}")
        return None
    except Exception as e:
        print(f"Error processing audio file {audio_file_path}: {str(e)}")
        return None

# Example usage (assuming you have a sample audio file path and a fitted scaler)
# if 'combined_df' in locals() and len(combined_df) > 0 and 'scaler_sequential' in locals():
#     sample_audio_path = combined_df['audio_path'].iloc[0]
#     print(f"\nTesting preprocess function with: {sample_audio_path}")
#     processed_input = preprocess_audio_for_sequential_model(sample_audio_path, scaler_sequential)
#     if processed_input is not None:
#         print("Preprocessing successful. Output shape:", processed_input.shape)
#     else:
#         print("Preprocessing failed.")
# else:
#     print("\nSkipping example usage: combined_df or scaler_sequential not found.")



# Assuming these are loaded elsewhere in your Streamlit app:
# arousal_model = tf.keras.models.load_model('path/to/your/arousal_model.keras')
# valence_model = tf.keras.models.load_model('path/to/your/valence_model.keras')
# scaler_sequential = joblib.load('path/to/your/scaler_sequential.pkl') # Load your fitted scaler
# AROUSAL_CLASSES = ['Low', 'Medium', 'High'] # Define these
# VALENCE_CLASSES = ['Negative', 'Neutral', 'Positive'] # Define these
# determine_emotion_from_text function (as defined previously)

# Re-define the determine_emotion_from_text function for clarity in this snippet
def determine_emotion_from_text(arousal_text, valence_text):
    """Determines emotion based on text arousal and valence labels."""
    if arousal_text == 'Low':
        if valence_text == 'Negative':
            return 'Whining'
        elif valence_text == 'Neutral':
            return 'Sigh'
        elif valence_text == 'Positive':
            return 'Happy'
    elif arousal_text == 'Medium':
        if valence_text == 'Negative':
            return 'Frustrated'
        elif valence_text == 'Neutral':
            return 'Alert'
        elif valence_text == 'Positive':
            return 'Playful'
    elif valence_text == 'Negative': # Assuming High arousal negative is Aggressive
        return 'Aggressive'
    elif valence_text == 'Neutral': # Assuming High arousal neutral is Neutral
        return 'Neutral'
    elif valence_text == 'Positive': # Assuming High arousal positive is Excited
        return 'Excited'
    else:
        return 'Unknown' # Should not happen with valid inputs

# Assuming the preprocess_audio_for_sequential_model function is also defined elsewhere
# from the previous code cell.
def split_audio_into_chunks(audio_path, chunk_duration_sec=2.0, sr=SR):
    y, _ = librosa.load(audio_path, sr=sr)
    total_duration = librosa.get_duration(y=y, sr=sr)
    chunk_length = int(sr * chunk_duration_sec)
    chunks = []

    for start in range(0, len(y), chunk_length):
        end = start + chunk_length
        chunk = y[start:end]
        if len(chunk) < chunk_length:
            padding = np.zeros(chunk_length - len(chunk))
            chunk = np.concatenate([chunk, padding])
        chunks.append(chunk)

    return chunks

uploaded_file = st.file_uploader("Upload a dog bark (.wav) file", type=["wav"])
scaler_sequential = joblib.load('scaler_sequential.pkl')

if uploaded_file:
    st.audio(uploaded_file)

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # --- Use the preprocess function for sequential models ---
    # IMPORTANT: Ensure arousal_model, valence_model, and scaler_sequential are loaded before this point
    # Also ensure preprocess_audio_for_sequential_model and determine_emotion_from_text are defined
    AROUSAL_CLASSES = ['Low', 'Medium', 'High']
    VALENCE_CLASSES = ['Negative', 'Neutral', 'Positive']

    #processed_features = preprocess_audio_for_sequential_model(tmp_path, scaler_sequential)

    chunked_audio = split_audio_into_chunks(tmp_path)

    valence_scores = []
    arousal_scores = []
    emotion_labels = []

    for chunk in chunked_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
            sf.write(chunk_file.name, chunk, SR)
            chunk_path = chunk_file.name

        processed = preprocess_audio_for_sequential_model(chunk_path, scaler_sequential)

        if os.path.exists(chunk_path):
            os.remove(chunk_path)

        if processed is not None:
            arousal_pred = arousal_model.predict(processed)
            valence_pred = valence_model.predict(processed)

            arousal_idx = np.argmax(arousal_pred)
            valence_idx = np.argmax(valence_pred)

            arousal_text = AROUSAL_CLASSES[arousal_idx]
            valence_text = VALENCE_CLASSES[valence_idx]
            emotion_text = determine_emotion_from_text(arousal_text, valence_text)

            # Display result
            # st.success(f"📈 Predicted Arousal: **{arousal_text}**")
            # st.success(f"📉 Predicted Valence: **{valence_text}**")
            # st.success(f"🐶 Predicted Emotion: **{emotion_text}**")

            arousal_scores.append(arousal_idx)
            valence_scores.append(valence_idx)
            emotion_labels.append(emotion_text)

    if emotion_labels:
        results_df = pd.DataFrame({
            #"Chunk": list(range(len(emotion_labels))),
            "Time Range": [f"{i*2}–{(i+1)*2} sec" for i in range(len(emotion_labels))],
            "Arousal": [AROUSAL_CLASSES[val] for val in arousal_scores], 
            "Valence": [VALENCE_CLASSES[val] for val in valence_scores],
            "Emotion": emotion_labels
        })
        st.subheader("📋 Emotion prediction every 2 seconds")
        st.dataframe(results_df)

        df = pd.DataFrame({
            # "Chunk": list(range(1, len(chunked_audio)+1)),
            #"Chunk": list(range(1, len(valence_scores)+1)),
            #"Chunk": list(range(len(valence_scores))), 
            "Time Range": [f"{i*2}–{(i+1)*2}s" for i in range(len(valence_scores))],
            "Valence": valence_scores,
            "Arousal": arousal_scores
        })
        df.set_index("Time Range", inplace=True)

        st.subheader("📊 Emotion Over Time")
        st.line_chart(df)
        st.caption("🔹 Arousal scale: 0 = Low, 1 = Medium, 2 = High")
        st.caption("🔹 Valence scale: 0 = Negative, 1 = Neutral, 2 = Positive")

        st.subheader("🌈 Emotion Over Time (Area Chart)")
        st.area_chart(df)


            # color_map = {
            #     "Excited": "gold", "Aggressive": "red", "Whining": "orange",
            #     "Alert": "deepskyblue", "Sigh": "lightgrey", "Playful": "violet",
            #     "Neutral": "gray", "Frustrated": "darkorange", "Happy": "green"
            # }
            # fig, ax = plt.subplots(figsize=(10, 1))
            # for i, mood in enumerate(emotion_labels):
            #     ax.barh(0, 1, left=i, color=color_map.get(mood, "black"))
            # ax.set_xlim(0, len(emotion_labels))
            # ax.set_yticks([])
            # ax.set_xticks(range(len(emotion_labels)))
            # ax.set_xticklabels([f"C{i+1}" for i in range(len(emotion_labels))], rotation=45)
            # ax.set_title("Detected Emotions Across Audio Chunks")
            # st.pyplot(fig)


    # Clean up the temporary file
    os.remove(tmp_path)

    # if processed_features is not None:
    #     try:
    #         # Predict using the Arousal and Valence models
    #         arousal_predictions_onehot = arousal_model.predict(processed_features)
    #         valence_predictions_onehot = valence_model.predict(processed_features)

    #         # Get the predicted class index (argmax)
    #         arousal_pred_encoded = np.argmax(arousal_predictions_onehot, axis=1)[0]
    #         valence_pred_encoded = np.argmax(valence_predictions_onehot, axis=1)[0]

    #         # Convert predicted indices back to text labels
    #         arousal_pred_text = AROUSAL_CLASSES[arousal_pred_encoded]
    #         valence_pred_text = VALENCE_CLASSES[valence_pred_encoded]

    #         # Determine the combined emotion label
    #         predicted_combined_emotion = determine_emotion_from_text(arousal_pred_text, valence_pred_text)

    #         # Display result
    #         st.success(f"📈 Predicted Arousal: **{arousal_pred_text}**")
    #         st.success(f"📉 Predicted Valence: **{valence_pred_text}**")
    #         st.success(f"🐶 Predicted Emotion: **{predicted_combined_emotion}**")

    #     except Exception as e:
    #         st.error(f"An error occurred during prediction: {e}")

    # else:
    #     st.error("Audio preprocessing failed. Cannot predict.")
    
else:
    st.info("Please upload a .wav file to get prediction.")

# file uploaded method end




# if uploaded_file:
#     # Show audio player
#     st.audio(uploaded_file)

#     # Dummy prediction
#     st.info("This is a test prototype – no prediction logic yet.")
#     st.success("🐶 Predicted emotion: **(placeholder)**")
# else:
#     st.info("Please upload a .wav file to see the demo.")

# def split_audio_into_chunks(audio_path, chunk_duration_sec=2.0, sr=SR):
#     y, _ = librosa.load(audio_path, sr=sr)
#     total_duration = librosa.get_duration(y=y, sr=sr)
#     chunk_length = int(sr * chunk_duration_sec)
#     chunks = []

#     for start in range(0, len(y), chunk_length):
#         end = start + chunk_length
#         chunk = y[start:end]
#         if len(chunk) < chunk_length:
#             padding = np.zeros(chunk_length - len(chunk))
#             chunk = np.concatenate([chunk, padding])
#         chunks.append(chunk)

#     return chunks

# chunked_audio = split_audio_into_chunks(tmp_path)

# valence_scores = []
# arousal_scores = []
# emotion_labels = []

# for chunk in chunked_audio:
#     # Save chunk to temp file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
#         #librosa.output.write_wav(chunk_file.name, chunk, sr=SR)
#         sf.write(chunk_file.name, chunk, SR)
#         processed = preprocess_audio_for_sequential_model(chunk_file.name, scaler_sequential)
#         os.remove(chunk_file.name)

#     if processed is not None:
#         arousal_pred = arousal_model.predict(processed)
#         valence_pred = valence_model.predict(processed)

#         arousal_idx = np.argmax(arousal_pred)
#         valence_idx = np.argmax(valence_pred)

#         arousal_text = AROUSAL_CLASSES[arousal_idx]
#         valence_text = VALENCE_CLASSES[valence_idx]

#         emotion_text = determine_emotion_from_text(arousal_text, valence_text)

#         arousal_scores.append(arousal_idx - 1)   # Map High=2, Medium=1, Low=0 to +1, 0, -1
#         valence_scores.append(valence_idx - 1)   # Same mapping
#         emotion_labels.append(emotion_text)

# st.subheader("📊 Emotion Over Time")

# # Line charts
# df = pd.DataFrame({
#     "Chunk": list(range(1, len(chunked_audio)+1)),
#     "Valence": valence_scores,
#     "Arousal": arousal_scores
# })
# st.line_chart(df.set_index("Chunk"))

# # Mood timeline bar
# color_map = {
#     "Excited": "gold", "Aggressive": "red", "Whining": "orange",
#     "Alert": "deepskyblue", "Sigh": "lightgrey", "Playful": "violet",
#     "Neutral": "gray", "Frustrated": "darkorange", "Happy": "green"
# }
# fig, ax = plt.subplots(figsize=(10, 1))
# for i, mood in enumerate(emotion_labels):
#     ax.barh(0, 1, left=i, color=color_map.get(mood, "black"))
# ax.set_xlim(0, len(emotion_labels))
# ax.set_yticks([])
# ax.set_xticks(range(len(emotion_labels)))
# ax.set_xticklabels([f"C{i+1}" for i in range(len(emotion_labels))], rotation=45)
# ax.set_title("Detected Emotions Across Audio Chunks")
# st.pyplot(fig)

# os.remove(tmp_path)