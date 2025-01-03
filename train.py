
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
import keras
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
from keras import layers
from sklearn.model_selection import train_test_split
import string
import re

batch_size = 32

# Load Sentiment140 Data
def load_sentiment140_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1', header=None)
    df = df[[0, 5]]
    df.columns = ['sentiment', 'text']
    
    # Convert sentiment to binary (0 = negative, 1 = positive)
    df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 4 else 0)
    
    return df

file_path = 'sentiment.csv'
df = load_sentiment140_data(file_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Custom standardization function
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

# Model constants
max_features = 20000
embedding_dim = 128
sequence_length = 500

# Text vectorization layer
vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)
# Function to remove stopwords
def remove_stopwords(text):
    stop_words = set(spacy.lang.en.stop_words.STOP_WORDS)
    return " ".join([word for word in text.split() if word not in stop_words])

# Function to lemmatize text using SpaCy
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Preprocess text: remove stopwords and lemmatize
def preprocess_text(text):
    text = remove_stopwords(text)  # Remove stopwords
    text = lemmatize_text(text)     # Lemmatize text
    return text

# Adapt the vectorize layer
text_ds = tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size)
vectorize_layer.adapt(text_ds)

# Vectorize text function
def vectorize_text(text, label):
    return vectorize_layer(text), label  # Remove unnecessary expand_dims

# Create TensorFlow datasets
def create_dataset(X, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.map(vectorize_text, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(10000).batch(batch_size).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

train_ds = create_dataset(X_train, y_train, batch_size)
val_ds = create_dataset(X_test, y_test, batch_size)

# Define the model
inputs = keras.Input(shape=(sequence_length,), dtype="int64")  # Ensure the correct input shape
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = keras.Model(inputs, predictions)

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
epochs = 3
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Save the model
model.save('sentiment_model.h5')  # Save in HDF5 format
# Alternatively, you can save in TensorFlow SavedModel format
# model.save('sentiment_model')

# End-to-end model for raw input
inputs = keras.Input(shape=(1,), dtype="string")
indices = vectorize_layer(inputs)
outputs = model(indices)
end_to_end_model = keras.Model(inputs, outputs)

# Compile end-to-end model
end_to_end_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Evaluate on raw test data
raw_test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
end_to_end_model.evaluate(raw_test_ds)

# Load the model (if needed)
# loaded_model = keras.models.load_model('sentiment_model.h5')
