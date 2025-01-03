import os
import pandas as pd
import tensorflow as tf
from keras import layers
import keras
import string
import re  # Import the re module for regular expressions

# Load the saved model
loaded_model = keras.models.load_model('sentiment_model.h5')

# Load the sentiment data to recreate the vectorization layer
def load_sentiment140_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1', header=None)
    df = df[[0, 5]]
    df.columns = ['sentiment', 'text']
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 4 else 0)
    return df

# Load the data to adapt the vectorization layer
file_path = 'sentiment.csv'  # Change this to your CSV file path
df = load_sentiment140_data(file_path)

# Custom standardization function
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

# Model constants
max_features = 20000
sequence_length = 500

# Recreate the TextVectorization layer
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# Adapt the vectorize layer using the training data
text_ds = tf.data.Dataset.from_tensor_slices(df['text']).batch(32)
vectorize_layer.adapt(text_ds)

# Function to preprocess and predict sentiment
def predict_sentiment(text):
    # Create a TensorFlow tensor from the input text
    input_data = tf.constant([text])
    
    # Vectorize the input text
    vectorized_text = vectorize_layer(input_data)
    
    # Make a prediction
    prediction = loaded_model.predict(vectorized_text)
    
    # Convert prediction to binary sentiment
    sentiment = 1 if prediction[0] > 0.5 else 0  # Threshold of 0.5 for binary classification
    return sentiment, prediction[0][0]  # Return sentiment and prediction probability

# Sample input for testing
sample_texts = [
    "ita quite a bad day",
    "Its quite a good day",
    "The mobile is really amazing",
    "The mobile is really irritating"
]

# Test the model with sample inputs
for text in sample_texts:
    sentiment, probability = predict_sentiment(text)
    print(f"Text: {text}\nPredicted Sentiment: {'Positive' if sentiment == 1 else 'Negative'}, Probability: {probability:.4f}\n")
