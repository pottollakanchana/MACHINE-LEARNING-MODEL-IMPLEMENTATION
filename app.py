import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Spam Classifier", page_icon="📧")

# Title
st.title("📧 Email / SMS Spam Detector")
st.write("Type a message below and check whether it's spam or not.")

# Input box
msg = st.text_area("✍️ Enter your message:")

# Button
if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message first.")
    else:
        data = vectorizer.transform([msg]).toarray()
        result = model.predict(data)

        if result[0] == 1:
            st.error("🚨 This is SPAM message!")
        else:
            st.success("✅ This is NOT spam")