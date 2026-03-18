import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

df = load_data()

st.title("📧 Email Spam Detection")

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

st.subheader("Spam vs Not Spam")

plt.figure()
sns.countplot(x='label', data=df)
st.pyplot(plt)

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
accuracy = accuracy_score(y_test, y_pred)

st.write("Accuracy:", accuracy)

st.subheader("Check Your Email")

user_input = st.text_area("Enter Message")

if st.button("Predict"):
    input_vec = vectorizer.transform([user_input])
    result = model.predict(input_vec)

    if result[0] == 1:
        st.error("🚫 Spam")
    else:
        st.success("✅ Not Spam")
