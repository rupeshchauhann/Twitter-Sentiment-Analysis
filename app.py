import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.corpus import stopwords
import plotly.express as px

# Load the saved models
vectorizer = joblib.load('vectorizer.joblib')
column_names = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

loaded_models = {}
for label in column_names:
    loaded_models[label] = joblib.load(f"./models each label/{label}_model.joblib")

# Preprocess the input text
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in text.split() if token.lower() not in stop_words]
    text = ' '.join(tokens)
    return text

# User input classification and plotly graph
def classify_and_show_graph(input_text):
    input_text = preprocess_text(input_text)
    X_user_input_vectorized = vectorizer.transform([input_text])

    user_predictions = {}
    for label, model in loaded_models.items():
        prediction = model.predict_proba(X_user_input_vectorized)[0][1]  # Get the probability of the positive class
        user_predictions[label] = prediction

    # Create a DataFrame from the user predictions
    df = pd.DataFrame.from_dict(user_predictions, orient='index', columns=['Probability'])
    df_sorted = df.sort_values(by='Probability', ascending=False)

    # Show the sorted probabilities in a table
    st.table(df_sorted)

    # Plot the graph using Plotly
    fig = px.bar(df_sorted, x=df_sorted.index, y='Probability', labels={'x': 'Emotion', 'y': 'Probability'})

    # Show the Plotly graph
    st.plotly_chart(fig)


# Streamlit app
def main():
    # Set the page title
    st.title("Twitter Emotion Classification")

    # User input
    input_text = st.text_input("Enter Twitter comment")

    # Classify and show graph button
    if st.button("Predict"):
        if input_text:
            classify_and_show_graph(input_text)
        else:
            st.warning("Please enter some text.")


# Run the Streamlit app
if __name__ == "__main__":
    main()
