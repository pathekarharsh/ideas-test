import streamlit as st
import requests
from typing import Dict, Any
import re

API_URL = "https://ideas-test.onrender.com"

# Configure the page
st.set_page_config(
    page_title="College Admission Chatbot",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS for chat bubbles
st.markdown("""
    <style>
    .user-bubble {
        background-color: #2b313e;
        color: #fff;
        padding: 1rem;
        border-radius: 1rem 1rem 0 1rem;
        margin-bottom: 0.5rem;
        max-width: 70%;
        align-self: flex-end;
    }
    .bot-bubble {
        background-color: #475063;
        color: #fff;
        padding: 1rem;
        border-radius: 1rem 1rem 1rem 0;
        margin-bottom: 0.5rem;
        max-width: 70%;
        align-self: flex-start;
    }
    .source-box {
        font-size: 0.85em;
        color: #b0b0b0;
        margin-top: 0.5rem;
        background: #23272f;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper to strip HTML tags
HTML_TAG_RE = re.compile(r'<[^>]+>')
def strip_html(text):
    return HTML_TAG_RE.sub('', text)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("🎓 College Admission Chatbot")
st.markdown("""
    Welcome! Ask me anything about VNIT admissions (UG, PG, MTech, PhD, etc). I'll answer using official documents.
""")

# Display chat history using st.chat_message
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(strip_html(message["content"]))
    else:
        with st.chat_message("assistant"):
            st.markdown(strip_html(message["content"]))
            if message.get("sources"):
                st.markdown(f"**Sources:** {', '.join(message['sources'])}", help="Source documents used for this answer.")

# Chat input
user_input = st.chat_input("Ask a question about admissions:")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.markdown(strip_html(user_input))
    
    # Get bot response
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": user_input}
        )
        response_data = response.json()
        
        # Sanitize backend answer
        answer = strip_html(response_data["answer"])
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": response_data.get("sources", [])
        })
        with st.chat_message("assistant"):
            st.markdown(answer)
            if response_data.get("sources"):
                st.markdown(f"**Sources:** {', '.join(response_data['sources'])}", help="Source documents used for this answer.")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure the backend server is running.")

# Add a clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun() 