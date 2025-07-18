# 🐶 Dog Bark Emotion Classifier (Prototype)

A **Streamlit-based prototype web app** that analyzes dog barks to detect emotion, valence, and arousal over time.  
Built for hackathon demo purposes using a public dataset from Barkopedia.

---

## 🎯 **Project Overview**

This prototype lets users **upload an audio clip** of a dog bark (`.wav`).  
If the clip is longer than 5 seconds, it is **split into multiple 2-second chunks**.  
Each chunk is then analyzed using two LSTM models:

- **Valence classifier** → predicts positivity: *Negative, Neutral, Positive*
- **Arousal classifier** → predicts excitement: *Low, Medium, High*

These are then mapped to one of nine possible emotions:

> *Whining, Sigh, Happy, Anxious, Alert, Playful, Aggressive, Neutral, Excited*

Finally, the app uses **Google Gemini** to suggest possible causes & calming tips for the dog based on the detected dominant emotion.

---

## 📊 **Visualizations**

For each uploaded clip, the app displays:

- 📋 **Table** showing the predicted emotion every 2 seconds
- 📈 **Emotion timeline chart** (line chart)
- 📊 **Confidence gauges** showing average model confidence
- 💡 **Gemini AI suggestions**: context‑aware tips based on the dominant emotion

---

## ⚙️ **How it works**

- Python **3.12**  
- Frontend & server: **Streamlit**
- Deep learning: **Keras** (TensorFlow backend)
- Audio processing: **librosa** (MFCC extraction)
- Visualization: **Plotly**
- AI suggestion: **Gemini API** (Google Generative AI)
- Dataset: Barkopedia  
  [→ View on Hugging Face](https://huggingface.co/spaces/ArlingtonCL2/BarkopediaDogEmotionClassification)

---

## 🚀 **Setup & Running Locally**

> ⚠ Requires Python **3.12**  
> Recommended to install separately from system Python (e.g., via Homebrew on macOS)

### 🐍 Create and activate virtual environment
```bash
# macOS example
brew install python@3.11

# Create & activate
# Replace the following below with your actual python installation path
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate

#OR On Windows
venv/Scripts/activate

```
  ## Install dependencies
```bash

pip install -r requirements.txt

```
  ## Setup Gemini API Key
  Create a .streamlit/secrets.toml file in your project root (If you're running the project locally):
```bash

# .streamlit/secrets.toml
GOOGLE_API_KEY = "your_gemini_api_key_here"

```

```bash

streamlit run index.py

```

