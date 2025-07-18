import streamlit as st
import joblib
import numpy as np
import librosa
import tempfile
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import soundfile as sf
import google.generativeai as genai # Import the Gemini library
from collections import Counter # To find the most common emotion
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageOps

emotion_images = {
    "Alert": "images/alert.jpg",
    "Sigh": "images/sigh.jpg",
    "Happy": "images/happy.jpg",
    "Aggressive": "images/aggressive.jpg",
    "Playful": "images/playful.jpg",
    "Whining": "images/whining.jpg",
    "Frustrated": "images/frustrated.jpg",
    "Neutral": "images/neutral.jpg",
    "Excited": "images/excited.jpg"
}

def load_cropped_image(path, size=(250, 250)):
    img = Image.open(path)
    cropped_img = ImageOps.fit(img, size, method=Image.LANCZOS, centering=(0.5, 0.5))
    return cropped_img

st.set_page_config(page_title="PawSound AI", layout="wide")
st.title("🐶 Dog Bark Emotion Classifier (Prototype)")



@st.cache_resource
def load_arousal_model():
    return keras.models.load_model("final_arousal_model.keras")

arousal_model = load_arousal_model()

@st.cache_resource
def load_valence_model():
    return keras.models.load_model("final_valence_model.keras")

valence_model = load_valence_model()

@st.cache_resource
def load_scaler_sequential():
    # Ensure this path is correct for your scaler file
    return joblib.load('scaler_sequential.pkl')

scaler_sequential = load_scaler_sequential()

# Define the parameters used for feature extraction and processing
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 1 # Crucial: Match the n_mfcc used for the sequential models
SEQUENCE_LENGTH = 25 # Crucial: Match the sequence length used for the sequential models

# Define Arousal and Valence Classes
AROUSAL_CLASSES = ['Low', 'Medium', 'High']
VALENCE_CLASSES = ['Negative', 'Neutral', 'Positive']


# Re-define the determine_emotion_from_text function for clarity
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
            return 'Frustrated' # Changed from 'Frustrated' to 'Anxious' for consistency with EMOTIONS list for Gemini
        elif valence_text == 'Neutral':
            return 'Alert'
        elif valence_text == 'Positive':
            return 'Playful'
    elif arousal_text == 'High': # Explicitly handle High arousal
        if valence_text == 'Negative':
            return 'Aggressive'
        elif valence_text == 'Neutral':
            return 'Neutral'
        elif valence_text == 'Positive':
            return 'Excited' # Your prompt used 'Excited' for high arousal positive
    return 'Unknown' # Should not happen with valid inputs

# --- Gemini API Configuration ---
# Define the prompt for Gemini API
gemini_prompt_template = """
You are an AI assistant specializing in dog behavior and well-being, providing information for veterinarians.
If the emotion is negative, Provide suggestions (In one short sentence) related to:
-What could be the cause
-What actions to soothe or calm the dog
If the emotion is generally considered positive or neutral (e.g., Happy, Playful, Alert, Sigh, Excited, Neutral), you can indicate that the dog may not require immediate behavioral intervention and is likely suitable for standard veterinary examination or procedures, but still provide any relevant general well-being tips if appropriate.
Do not give suggestions to go to the vet, because the user is most likely already a vet.

Here is the Dog Emotion: {emotion}
Please give your suggestions.

The output should be like
Emotion: <Insert emotion>

Sentence of suggestion
Format the output as a sentence that doesn't look too LLM generated.
"""

# Use st.cache_resource to load the Gemini model only once
@st.cache_resource
def load_gemini_model():
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
        st.success("Gemini API configured successfully.")
        return model
    except KeyError:
        st.error("Gemini API Key not found in Streamlit secrets. Please add it to .streamlit/secrets.toml")
        return None
    except Exception as e:
        st.error(f"Error initializing Gemini model: {e}")
        return None

gemini_model = load_gemini_model()

# Function to Generate Gemini Suggestions
def get_gemini_suggestions(predicted_emotion, model, prompt_template):
    """
    Generates suggestions from Gemini API based on the predicted emotion.
    Args:
        predicted_emotion (str): The predicted dog emotion (e.g., 'Whining', 'Happy').
        model (genai.GenerativeModel): The initialized Gemini API model.
        prompt_template (str): The template for the Gemini API prompt.
    Returns:
        str: Gemini suggestions, or an error message/standard message if API call fails
             or the emotion is positive/neutral.
    """
    if not model:
        return "Gemini model is not available to provide suggestions due to API key or initialization issues."

    try:
        prompt = prompt_template.format(emotion=predicted_emotion)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error fetching suggestions from Gemini API: {e}"


# --- End Gemini API Configuration ---




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
        st.error(f"Error: Audio file not found at {audio_file_path}")
        return None
    except Exception as e:
        st.error(f"Error processing audio file {audio_file_path}: {str(e)}")
        return None



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
# scaler_sequential = joblib.load('scaler_sequential.pkl') # Moved to @st.cache_resource

if uploaded_file:
    audio_data, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file)

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # --- Use the preprocess function for sequential models ---
    # IMPORTANT: Ensure arousal_model, valence_model, and scaler_sequential are loaded before this point
    # Also ensure preprocess_audio_for_sequential_model and determine_emotion_from_text are defined
    # AROUSAL_CLASSES = ['Low', 'Medium', 'High'] # Defined globally
    # VALENCE_CLASSES = ['Negative', 'Neutral', 'Positive'] # Defined globally

    #processed_features = preprocess_audio_for_sequential_model(tmp_path, scaler_sequential)

    chunked_audio = split_audio_into_chunks(tmp_path)

    valence_scores = [] # stores integer indices 0, 1, 2
    arousal_scores = [] # stores integer indices 0, 1, 2
    emotion_labels = [] # stores string labels like 'Happy', 'Whining'
    arousal_confidences = []
    valence_confidences = []

    with st.spinner("Analyzing audio chunks..."):
        for i, chunk in enumerate(chunked_audio):
            # Display progress (optional)
            # st.text(f"Processing chunk {i+1}/{len(chunked_audio)}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
                sf.write(chunk_file.name, chunk, SR)
                chunk_path = chunk_file.name

            processed = preprocess_audio_for_sequential_model(chunk_path, scaler_sequential)

            if os.path.exists(chunk_path):
                os.remove(chunk_path)

            if processed is not None:
                arousal_pred = arousal_model.predict(processed, verbose=0) # verbose=0 to suppress Keras output
                valence_pred = valence_model.predict(processed, verbose=0)

                arousal_idx = np.argmax(arousal_pred)
                valence_idx = np.argmax(valence_pred)

                arousal_confidences.append(float(arousal_pred[0][arousal_idx]) * 100)
                valence_confidences.append(float(valence_pred[0][valence_idx]) * 100)

                arousal_text = AROUSAL_CLASSES[arousal_idx]
                valence_text = VALENCE_CLASSES[valence_idx]
                emotion_text = determine_emotion_from_text(arousal_text, valence_text)
                
                arousal_scores.append(arousal_idx - 1)
                valence_scores.append(valence_idx - 1)
                emotion_labels.append(emotion_text)

            else:
                st.warning(f"Skipping chunk {i+1} due to preprocessing error.")
                # Append a placeholder or skip if processing failed for a chunk
                emotion_labels.append("Processing Error") # Or 'Unknown', 'N/A' etc.


    if arousal_confidences:
        arousal_confidence = np.mean(arousal_confidences)
    else:
        arousal_confidence = 0

    if valence_confidences:
        valence_confidence = np.mean(valence_confidences)
    else:
        valence_confidence = 0


    # Clean up the temporary file
    os.remove(tmp_path)

    if emotion_labels:
        # Start with the first emotion as the first transition
        transitions = [emotion_labels[0]] if emotion_labels else []

        # Loop from second element onward and compare to the previous
        for i in range(1, len(emotion_labels)):
            if emotion_labels[i] != emotion_labels[i - 1]:
                transitions.append(emotion_labels[i])

        st.subheader("🔁 Emotion Transitions")
        #st.write(" → ".join(transitions))
        cols = st.columns(len(transitions) * 2 - 1)  # extra cols for arrows

        for i, emotion in enumerate(transitions):
            with cols[i * 2]:
                img_path = emotion_images.get(emotion, "images/default.png")
                cropped_img = load_cropped_image(img_path)
                st.image(cropped_img, caption=emotion, use_container_width=False)
            if i < len(transitions) - 1:
                with cols[i * 2 + 1]:
                    #st.markdown("→", unsafe_allow_html=True)
                    st.markdown(
                        """
                        <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
                            <span style="font-size: 40px;">→</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )



        # --- Gemini AI Suggestions ---
        st.subheader("💡 Gemini AI Suggestions")
        if gemini_model:
            # Determine the most frequent emotion across all chunks
            # Exclude "Processing Error" from the count
            filtered_emotions = [e for e in emotion_labels if e != "Processing Error"]
            if filtered_emotions:
                overall_emotion = Counter(filtered_emotions).most_common(1)[0][0]
                with st.spinner(f"Generating AI suggestions for overall emotion: {overall_emotion}..."):
                    suggestions = get_gemini_suggestions(overall_emotion, gemini_model, gemini_prompt_template)
                st.info(suggestions)
            else:
                st.warning("No valid emotions were detected to generate suggestions.")
        else:
            st.warning("Gemini AI suggestions are not available because the model could not be loaded. Please check your API key.")




        # Create DataFrame for tabular display
        results_df = pd.DataFrame({
            #"Chunk": list(range(len(emotion_labels))),
            "Time Range": [f"{i*2}–{(i+1)*2} sec" for i in range(len(emotion_labels))],
            # "Arousal": [AROUSAL_CLASSES[val + 1] if isinstance(val, int) else val for val in arousal_scores], 
            # "Valence": [VALENCE_CLASSES[val + 1] if isinstance(val, int) else val for val in valence_scores],
            "Emotion": emotion_labels
        })
        st.subheader("📋 Emotion prediction every 2 seconds")
        st.dataframe(results_df)
    
        st.subheader("📋 Emotion Changes Over Time")
        time_ranges = [f"{i*2}–{(i+1)*2}s" for i in range(len(emotion_labels))]
        emotions = emotion_labels

        emotion_df = pd.DataFrame({
            "Time Range": time_ranges,
            "Emotion": emotions
        })

        # Plot line chart with categorical Y-axis
        fig = px.line(
            emotion_df,
            x="Time Range",
            y="Emotion",
            markers=True,
            #title="🐶 Emotion Changes Over Time",
        )

        fig.update_layout(
            yaxis_title="Emotion",
            xaxis_title="Time Range",
            xaxis_tickangle=0,
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    





        st.subheader("📈 Audio Waveform")
        downsampled = audio_data[::int(sr/500)]  # Downsample for performance
        st.line_chart(downsampled)

        #Confidence Scores
        ###############################################################################################################
        st.subheader("📊 Model Confidence Gauges")

        # Create 2 or 3 columns for side-by-side layout
        col1, col2 = st.columns(2)  # For 2 gauges
        # col1, col2, col3 = st.columns(3)  # Uncomment for 3 gauges

        with col1:
            st.subheader("Arousal Confidence")
            fig_arousal = go.Figure(go.Indicator(
                mode="gauge+number",
                value=arousal_confidence,   
                title={'text': "Arousal"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "yellow"},
                        {'range': [66, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            st.plotly_chart(fig_arousal, use_container_width=True)

        with col2:
            st.subheader("Valence Confidence")
            fig_valence = go.Figure(go.Indicator(
                mode="gauge+number",
                value=valence_confidence,
                title={'text': "Valence"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "skyblue"},
                        {'range': [66, 100], 'color': "green"}
                    ]
                }
            ))
            st.plotly_chart(fig_valence, use_container_width=True)
###############################################################################################################

    
else:
    st.info("Please upload a .wav file to get prediction.")
