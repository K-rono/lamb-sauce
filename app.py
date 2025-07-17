import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import time
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="PawSound AI", layout="wide")

# Emotion categories
EMOTIONS = [
    {"name": "Happy", "emoji": "😊"},
    {"name": "Alert", "emoji": "⚠️"},
    {"name": "Anxious", "emoji": "😰"},
    {"name": "Calm", "emoji": "😌"},
    {"name": "Playful", "emoji": "⚡"},
]

st.title("🐕 PawSound AI - Dog Emotion Recognition")

uploaded_file = st.file_uploader("Upload a dog sound file (wav, mp3, flac)", type=["wav", "mp3", "flac"])

if uploaded_file:
    audio_data, sr = librosa.load(uploaded_file, sr=None)
    st.audio(uploaded_file)
    st.success(f"Audio loaded: {len(audio_data)/sr:.2f}s at {sr}Hz")

    # Show waveform using line chart
    st.subheader("📈 Audio Waveform")
    downsampled = audio_data[::int(sr/1000)]  # Downsample for performance
    st.line_chart(downsampled)

    # Simulate classification
    if st.button("🔍 Analyze Emotion"):
        with st.spinner("Analyzing audio..."):
            time.sleep(2)  # Simulate processing
            top_emotion = random.choice(EMOTIONS)
            confidence = round(random.uniform(60, 95), 1)

        st.subheader("🎯 Emotion Result")
        st.markdown(f"### {top_emotion['emoji']} **{top_emotion['name']}**")
        st.progress(confidence / 100)
        st.write(f"Confidence Score: **{confidence}%**")

        # Show emotion result
        st.subheader("🎯 Emotion Result")
        st.markdown(f"### {top_emotion['emoji']} **{top_emotion['name']}**")
        
        # Gauge chart using Plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence Level"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ What This Does"):
        st.markdown("""
        - Upload a dog's sound (bark, whimper, etc.)
        - The app extracts basic audio features
        - Simulates AI-based emotion recognition
        - Shows estimated emotion + confidence
        """)

else:
    st.info("Upload a file to get started.")
