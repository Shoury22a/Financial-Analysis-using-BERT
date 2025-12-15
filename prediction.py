import streamlit as st
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import numpy as np
import os

model_save_dir = "financial_sentiment_model"
weights_path = os.path.join(model_save_dir, "bert_weights.h5")

st.title("📈 Financial Sentiment Analysis")
st.write("Enter financial news to analyze sentiment.")

@st.cache_resource
def load_model():
    model = TFBertForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
    model.load_weights(weights_path)
    tokenizer = BertTokenizer.from_pretrained(model_save_dir)
    return model, tokenizer

model, tokenizer = load_model()

label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text, threshold=0.65):
    encodings = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="tf")
    logits = model(encodings.data)[0]  
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]  

    max_prob = np.max(probs)
    sentiment = label_map[np.argmax(probs)] if max_prob >= threshold else "Neutral"

    return sentiment, {label_map[i]: round(float(probs[i]), 2) for i in range(len(label_map))}

user_input = st.text_area("📝 Enter financial news here:", "")
if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment, probs = predict_sentiment(user_input)
        st.write(f"**Prediction:** {sentiment}")
        st.write(f"**Confidence:**")
        st.write(f"- **Negative:** {probs['Negative']:.2f}")
        st.write(f"- **Neutral:** {probs['Neutral']:.2f}")
        st.write(f"- **Positive:** {probs['Positive']:.2f}")
    else:
        st.warning("⚠️ Please enter some text.")
