# 🐶 Dog Bark Emotion Classifier (Prototype)

A Streamlit-based prototype web app that analyzes dog barks to detect emotion, valence, and arousal over time.  
Built for hackathon demo purposes using a public dataset from Barkopedia.

---

## 🎯 **Project Overview**

This prototype lets users **upload an audio clip** of a dog bark (in `.wav` format).  
If the clip is longer than 5 seconds, it is **split into multiple 2-second chunks**.  
Each chunk is then analyzed using three separate LSTM models:

- **Emotion classifier**: predicts one of 9 moods  
  (*excited, aggressive, whining, alert, sigh, playful, neutral, frustrated, happy*)
- **Valence classifier**: predicts overall positivity (*high / medium / low*)
- **Arousal classifier**: predicts excitement level (*high / medium / low*)

Predictions are then visualized to show how the dog's mood changes over time.

---

## 📊 **Visualizations**

For each uploaded clip, the app generates:

- 📈 **Valence over time** (line chart)  
- 📈 **Arousal over time** (line chart)  
- 🟥 **Mood timeline**: color-coded stacked bar showing detected emotions across chunks

*Valence and arousal are categorical outputs mapped to numeric scores (e.g., high=1, medium=0, low=-1) to display as line charts.*

---

## ⚙️ **How it works**

- Built in **Python 3.12** (virtualenv recommended)
- Frontend & server: **Streamlit** (deployed on Streamlit Cloud)
- Deep learning: **Keras LSTM models** (TensorFlow backend)
- Audio feature extraction: **librosa** (MFCCs)
- Dataset: Public dataset from Barkopedia  
  [→ View on Hugging Face](https://huggingface.co/spaces/ArlingtonCL2/BarkopediaDogEmotionClassification)

---

## 🚀 **Setup & Running Locally**

> Requires Python **3.12**.  
> Recommended: install Python 3.12 alongside system Python (e.g., via Homebrew) and use a virtual environment.

```bash
# Install Python 3.12 (macOS example)
brew install python@3.12

# Create & activate virtual environment
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run locally
streamlit run index.py
