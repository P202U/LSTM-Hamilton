import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = load_model('lstm-hamelton.keras')
with open('lstm-tokenizer.pkl', 'rb') as token:
    tokenizer = pickle.load(token)

# Predict the next word
def predict_next_word(model, tokenizer, text, max_input_len):
    """
    Predicts the next word given a text context.
    max_input_len should be the exact input length required by the model.
    """
    token_list = tokenizer.texts_to_sequences([text])[0]
    # Truncate to the maximum allowed input length
    if len(token_list) > max_input_len:
        token_list = token_list[-max_input_len:]

    # Pad the sequence to the fixed input length
    token_list = pad_sequences([token_list],
                               maxlen=max_input_len,
                               padding='pre')

    # Predict probabilities
    predicted = model.predict(token_list, verbose=0)
    
    # Get the index of the highest probability
    predicted_index = np.argmax(predicted, axis=1)[0]
    
    # Efficiently retrieve the word using index_word mapping
    if predicted_index in tokenizer.index_word:
        return tokenizer.index_word[predicted_index]

    return None

# Streamlit app
st.title('Next word prediction with hamilton dataset')
input_text = st.text_input('Enter sequence of words', 'Fran. For this releefe much')

if st.button('Predict next word'):
    max_input_length = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_input_length)
    if next_word:
        st.write(f'Next word: **{next_word}**')
    else:
        st.write('Could not predict a next word.')