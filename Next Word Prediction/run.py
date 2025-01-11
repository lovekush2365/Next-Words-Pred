import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox

def generate_text(tokenizer, model, max_sequence_len, seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_probs = model.predict(token_list)[0]
    output_words = []
    
    for _ in range(3):  # Generate 3 words
        predicted = np.random.choice(len(predicted_probs), p=predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        output_words.append(output_word)
        seed_text += " " + output_word  # Update seed_text for next prediction

        # Update token_list for the next prediction
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

        predicted_probs = model.predict(token_list)[0]

    return output_words

def load_model_and_tokenizer(language):
    if language == "Hindi":
        corpus_file = "Hindi_Model/new_corpus.txt"
        model_file = "Hindi_Model/hindiLstmModel2.h5"
    else:
        corpus_file = "Eng_Model/new_corpus.txt"
        model_file = "Eng_Model/englishLstmModel2.h5"

    with open(corpus_file, "r", encoding='utf-8') as file:
        corpus_text = file.read()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([corpus_text])
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in corpus_text.split('\n'):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(seq) for seq in input_sequences])

    model = load_model(model_file)

    return tokenizer, model, max_sequence_len

def generate():
    language = language_var.get()
    seed_text = seed_text_entry.get()

    if not seed_text:
        messagebox.showerror("Input Error", "Please enter a seed text.")
        return

    tokenizer, model, max_sequence_len = load_model_and_tokenizer(language)
    
    # Generate 3 words
    generated_words = generate_text(tokenizer, model, max_sequence_len, seed_text)

    output_text.delete(1.0, tk.END)  # Clear previous output
    for word in generated_words:
        output_text.insert(tk.END, f"{seed_text} {word}\n")  # Insert original and generated text on new lines

# GUI Setup
root = tk.Tk()
root.title("Next Word Prediction")

language_var = tk.StringVar(value="English")

# Language Selection
language_frame = tk.Frame(root)
language_frame.pack(pady=10)

hindi_radio = tk.Radiobutton(language_frame, text="Hindi", variable=language_var, value="Hindi")
hindi_radio.pack(side=tk.LEFT, padx=5)

english_radio = tk.Radiobutton(language_frame, text="English", variable=language_var, value="English")
english_radio.pack(side=tk.LEFT, padx=5)

# Seed Text Input
tk.Label(root, text="Enter Seed Text:").pack(pady=5)
seed_text_entry = tk.Entry(root, width=50)
seed_text_entry.pack(pady=5)

# Generate Button
generate_button = tk.Button(root, text="Generate", command=generate)
generate_button.pack(pady=20)

# Output Text Area
output_text = tk.Text(root, height=10, width=50)
output_text.pack(pady=5)

# Start the GUI loop
root.mainloop()
