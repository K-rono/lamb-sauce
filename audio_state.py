# audio_state.py
import queue

audio_buffer = queue.Queue()
current_prediction = {"label": "No prediction yet"}