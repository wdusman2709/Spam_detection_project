import numpy as np
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

df = load_data()
st.title("📧 Email Spam Detection App")
st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Dataset Info")
st.write(df['label'].value_counts())

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

st.subheader("Spam vs Not Spam")

plt.figure()
sns.countplot(x='label', data=df)
st.pyplot(plt)

fig = px.histogram(df, x="label")
st.plotly_chart(fig)

chart = alt.Chart(df).mark_bar().encode(
    x='label',
    y='count()'
)
st.altair_chart(chart)

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

st.subheader("Model Performance")

st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
st.pyplot(plt)

with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

st.subheader("Check Your Email")

user_input = st.text_area("Enter Email Message")

if st.button("Predict"):
    input_vec = vectorizer.transform([user_input])
    result = model.predict(input_vec)

    if result[0] == 1:
        st.error("🚫 Spam Email")
    else:
        st.success("✅ Not Spam Email")
