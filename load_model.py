from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# Set your Hugging Face token (optional for public models)
# Replace 'your-token-here' with your actual HF token if needed
HF_TOKEN = ""  # Leave as is for public models or replace with token

# Define the model name
model_name = "distilbert/distilbert-base-multilingual-cased"

# Set the cache directory
cache_dir = "./model_cache"

# Create cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)

# Download tokenizer
print(f"Downloading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN if HF_TOKEN else None,
    cache_dir=cache_dir
)

# Download model
print(f"Downloading model {model_name}...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    token=HF_TOKEN if HF_TOKEN else None,
    cache_dir=cache_dir
)

# Test the download
print("Testing the downloaded model...")
text = "This is a test sentence"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print("Model downloaded and tested successfully!")
print(f"Model and tokenizer saved to: {cache_dir}")