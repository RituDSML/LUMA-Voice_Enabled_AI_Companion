# emotion_model.py - LUMA's Emotion Detection
# Uses a locally fine-tuned BERT classifier (trained on GoEmotions,
# 4-group schema: positive/negative/ambiguous/neutral)

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# Path to your locally saved fine-tuned model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "luma_emotion_model")

# Load once at import time (not inside the function — loading is slow, ~1-2 sec)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # inference mode, disables dropout etc.

def detect_emotion(text):
    """Detect emotion from text using locally fine-tuned BERT model"""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        all_emotions = [
            {"label": model.config.id2label[i], "score": round(probs[i].item(), 3)}
            for i in range(len(probs))
        ]

        top = max(all_emotions, key=lambda x: x["score"])

        return {
            'emotion': top['label'],
            'confidence': top['score'],
            'all_emotions': all_emotions
        }

    except Exception as e:
        print(f"Emotion detection error: {e}")
        return {'emotion': 'neutral', 'confidence': 0.0, 'all_emotions': []}

def get_emotion_label(text):
    """Simple function returning just the emotion string"""
    result = detect_emotion(text)
    return result['emotion']

# Test
if __name__ == "__main__":
    test_sentences = [
        "I feel so anxious and worried about everything",
        "I am really happy today!",
        "I feel completely hopeless and sad",
        "I am so angry right now",
        "I don't feel anything"
    ]

    print("Testing LUMA Emotion Detection (local model)")
    print("=" * 40)
    for sentence in test_sentences:
        result = detect_emotion(sentence)
        print(f"Text: {sentence}")
        print(f"Emotion: {result['emotion']} (confidence: {result['confidence']})")
        print("-" * 40)