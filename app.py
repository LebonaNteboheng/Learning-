# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import webbrowser
from threading import Timer

app = Flask(__name__)


model_file = r'nextWord_model.h5'  
tokenizer_file = 'tokenizer.pickle'

# Function to create and save the tokenizer
def create_tokenizer():
    print("Creating tokenizer from training data..")
    
    try:
        with open('friends1.txt', 'r', encoding='utf-8') as file:
            corpus = file.read()
    except FileNotFoundError:
        raise FileNotFoundError("friends1.txt file not found. This file is needed to recreate the tokenizer.")
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([corpus])
    
    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return tokenizer

# This function determines the sequence length from the model
def get_sequence_length(model):
    # Try different approaches to get the sequence length
    try:
        # Check if the first layer is Embedding and has input_length attribute
        if hasattr(model.layers[0], 'input_length'):
            return model.layers[0].input_length + 1
    except:
        pass
    
    try:
        # Try to access the input shape directly
        input_shape = model.input_shape
        if input_shape and len(input_shape) > 1:
            return input_shape[1] + 1
    except:
        pass
    
    # Default to a common sequence length if we can't determine it
    print("Warning: Could not determine sequence length from model. Using default value of 4.")
    return 4  # Default value (3+1)

# Load the model and create or load tokenizer
def setup_prediction_environment():
    # Load the model
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found. Make sure the path is correct.")
        
    model = load_model(model_file)
    print("Model loaded successfully.")
    
    # Create or load tokenizer
    if os.path.exists(tokenizer_file):
        print("Loading existing tokenizer...")
        with open(tokenizer_file, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        print("Tokenizer not found. Creating new tokenizer...")
        tokenizer = create_tokenizer()
    
    # Get sequence length
    max_sequence_length = get_sequence_length(model)
    print(f"Using sequence length: {max_sequence_length}")
    
    return model, tokenizer, max_sequence_length

# Load the model and tokenizer
try:
    model, tokenizer, max_sequence_length = setup_prediction_environment()
except Exception as e:
    print(f"Error setting up prediction environment: {str(e)}")
    raise

# Function to predict next words
def predict_next_words(seed_text, next_words, model, tokenizer, max_sequence_length):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        
        # Make sure token_list has the right shape for prediction
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        
        seed_text += " " + output_word
    
    return seed_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    seed_text = request.form['seed_text']
    next_words = int(request.form['next_words'])
    
    if next_words <= 0:
        return jsonify({'error': 'Number of next words must be positive'})
    
    try:
        result = predict_next_words(seed_text, next_words, model, tokenizer, max_sequence_length)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

def open_browser():
    webbrowser.open_new('http://localhost:5000')

if __name__ == '__main__':
    # Only open browser if not in debug mode
    if not app.debug:
        Timer(1.5, open_browser).start()
    app.run(debug=False)