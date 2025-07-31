import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
import pickle
import streamlit as st
stop_words = set(stopwords.words("english"))





model = load_model("lstm_model(1).h5")



#function for removing the special characters and punctuation
def clean_text(text):
     # 1. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # 2. Remove special characters (but keep spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    #3 lowercase 
    text = text.lower()
    
    # 4. Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # 5. Strip leading and trailing spaces
    # text = text.strip()

    #6. remove Stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)


 
 
with open("tokenizer(1).pkl", "rb") as f:
    tokenizer = pickle.load(f)

maxlen = 1000
# text = ["India's Prime minister is Rahul Gandhi"]
# cleaned_text  = [clean_text(t) for t in text]
# tokenized_text = tokenizer.texts_to_sequences(cleaned_text)
# x = pad_sequences(tokenized_text, maxlen=maxlen)
# print((model.predict(x) > 0.7).astype(int) )

st.title("Fake News Detector")
st.write("Enter a news article below to check whether it is Fake or Real")

news_input = st.text_area("News Article", "")

cleaned_input = clean_text(news_input)


if st.button("Check News"):
  if cleaned_input.strip():
    transform_input = tokenizer.texts_to_sequences([cleaned_input])
    transform_input = pad_sequences(transform_input, maxlen=maxlen)
    prediction = (model.predict(transform_input) > 0.6).astype(int)
    print(model.predict(transform_input))

    if prediction[0] == 0:
      st.error("The News is Fake!")
    else: 
      st.success("The News is Real!")
  else:
    st.warning("Please enter some text to analyze. ")

