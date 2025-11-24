from langchain_helper import get_response_from_query, create_vector_db_from_youtube_url
import streamlit as st
import textwrap

# Run streamlit with `streamlit run main.py`
# Example URL - https://www.youtube.com/watch?v=zjkBMFhNj_g
# Example Question - What is jailbreak and give examples based on video 

st.title("Youtube Assistant") # open port 8051

with st.sidebar:
    with st.form(key="my_form"):
        yt_url = st.sidebar.text_area(
            label = "What is Youtube video url ?",
            max_chars=50
        )
        query = st.sidebar.text_area(
            label = "Ask me about the video ?",
            max_chars=100,
            key = "query"
        )

        submit_button = st.form_submit_button(label="Submit")

if query and yt_url:
    db = create_vector_db_from_youtube_url(yt_url)
    response, docs = get_response_from_query(db, query)
    st.subheader("Answer: ")
    st.text(textwrap.fill(response, width = 80))