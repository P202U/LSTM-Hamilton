# Hamilton Next Word Predictor

This is a live demo of a next-word prediction model trained on the complete works of Alexander Hamilton. The application uses a Long Short-Term Memory (LSTM) neural network to suggest the most likely next word based on the text you provide.

## How It Works

The model was trained on a large dataset of Hamilton's writings, allowing it to learn the patterns, grammar, and vocabulary of his unique style.

- **Enter Text:** Type a sentence or a phrase into the input box.
- **Predict:** The model takes the last few words of your input as context.
- **Generate:** It uses the learned patterns to predict the single word that is most likely to follow.

This is a fun and simple example of a word-level language model.

## Technologies Used

- **Python:** The core programming language.
- **Streamlit:** For creating the interactive web application.
- **TensorFlow/Keras:** For building and loading the LSTM neural network model.
- **Numpy:** For numerical operations on the model's output.
- **Pickle:** For loading the pre-trained tokenizer.

## Live Demo

You can try the Hamilton Next Word Predictor right now by clicking the link below:

[LSTM Hamilton](https://lstm-hamilton.streamlit.app/)

**Note:** The model's predictions are based on its training data and may not always be grammatically perfect or contextually appropriate outside of the Hamilton-esque style it learned.
