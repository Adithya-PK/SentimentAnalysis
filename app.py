import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


st.title("Multilingual Sentiment Analysis App")
st.write("Enter text below to analyze its sentiment")

import streamlit as st

# Create sidebar
with st.sidebar:
    # Add "About" title
    st.title("About")
    
    # Model description
    st.write("")
    st.write("This model is a fine-tuned version of distilbert/distilbert-base-multilingual-cased for multilingual sentiment analysis. It leverages synthetic data from multiple sources to achieve robust performance across different languages and cultural contexts.")
    st.write("")
   
    # Supported languages in smaller text using markdown
    st.markdown("""
    <small>Supports English plus Chinese (中文), Spanish (Español), Hindi (हिन्दी), Arabic (العربية), Bengali (বাংলা), Portuguese (Português), Russian (Русский), Japanese (日本語), German (Deutsch), Malay (Bahasa Melayu), Telugu (తెలుగు), Vietnamese (Tiếng Việt), Korean (한국어), French (Français), Turkish (Türkçe), Italian (Italiano), Polish (Polski), Ukrainian (Українська), Tagalog, Dutch (Nederlands), Swiss German (Schweizerdeutsch)</small>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_name = "tabularisai/multilingual-sentiment-analysis"
    cache_dir = "./model_cache"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

nlp_pipeline = load_model()


user_input = st.text_area("Enter your text here:", "Type something to analyze...")

if st.button("Analyze Sentiment"):
    if user_input.strip():

        with st.spinner("Analyzing..."):
            result = nlp_pipeline(user_input)
        

        st.subheader("Results:")
        sentiment = result[0]['label']
        confidence = result[0]['score']
        

        if sentiment.lower() == "positive":  # Case-insensitive check
             st.success(f"Sentiment: {sentiment} (Confidence: {confidence:.2%})")  # Green box
        elif sentiment.lower() == "negative":
             st.error(f"Sentiment: {sentiment} (Confidence: {confidence:.2%})")    # Red box
        else:
             st.warning(f"Sentiment: {sentiment} (Confidence: {confidence:.2%})")   # Yellow for neutral/other

        st.write(f"The model predicts this text has a {sentiment.lower()} sentiment with {confidence:.2%} confidence.")
    else:
        st.warning("Please enter some text to analyze!")
