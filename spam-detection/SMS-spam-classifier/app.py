import streamlit as st
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import string
nltk.download('stopwords')
nltk.download('punkt')
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_msg = st.text_input("Enter the Message")

def transform_text(text):
    text = text.lower() # lower case
    text = nltk.word_tokenize(text) # separating words
    
    # removing unnecessary words and punctuation 
    y = []
    for i in text:
        if i.isalnum() and i not in stopwords.words('english') \
            and i not in string.punctuation:
            y.append(ps.stem(i))

    
    return " ".join(y)
if st.button('Predict'):
    transformed_msg = transform_text(input_msg)

    vector_msg = tfidf.transform([transformed_msg])

    result = model.predict(vector_msg)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header('Not Spam')