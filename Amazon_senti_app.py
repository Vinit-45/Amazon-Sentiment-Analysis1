# %%
import streamlit as st
import joblib

# %%
model = joblib.load('logistic.pkl')
tfidf = joblib.load('tfidf_vect.pkl')
le = joblib.load('label_encode.pkl')

# %%
def predict_sentiment(text):
    text_vector = tfidf.transform([text])
    pred = model.predict(text_vector)
    sentiment=le.inverse_transform(pred)[0]
    return sentiment

# %%
st.set_page_config(page_title='Sentiment predictor',layout='centered')
st.title('📦 Amazon Sentiment Analysis')
st.write("Enter a product review to  predict if it's **Positive**, **Neutral**, or **Negative** ")

user_input = st.text_area('Enter your review')

if st.button('Predict Sentiment'):
    if not user_input.strip():
        st.warning('Please enter some text to analyze')
    else:
        result = predict_sentiment(user_input)
        st.success(f'Predicted Sentiment: {result}')

# %%



