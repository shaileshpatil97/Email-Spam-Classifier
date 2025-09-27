import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
nltk.download('stopwords')

ps = PorterStemmer()

tfidf = pickle.load(open('models/vectorizer.pkl', 'rb'))
model = pickle.load(open('models/model.pkl', 'rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title('Email Spam Classifier')
input_email = st.text_area('Enter the email text:')

if st.button("Classify"):
    # Step 1: Preprocess
    processed_text = transform_text(input_email)

    # Step 2: Vectorize
    vector_input = tfidf.transform([processed_text])

    # Step 3: Predict
    result = model.predict(vector_input)[0]

    # Step 4: Show result
    if result == 1:
        st.error('🚫 This email is **SPAM**.')
    else:
        st.success('✅ This email is **NOT SPAM**.')