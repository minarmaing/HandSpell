import streamlit as st
import requests
import os

st.set_page_config(page_title="HandSpell Tester UI", layout="centered")
st.title("✋ HandSpell Basic Tester")
st.write("Upload a 28x28 grayscale hand sign image to validate the Docker API.")

uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    response = requests.post(
        os.getenv("API_URL", "http://localhost:8001/predict"),
        files={"file": uploaded_file},
    )
    if response.status_code == 200:
        response_json = response.json()
        st.success(f"Predicted letter: {response_json['letter']}")
    else:
        st.error(f"Error {response.status_code}: {response.text}")
