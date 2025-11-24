# Generate pet names
# Run streamlit with `streamlit run main.py`

from langchain_helper import generate_pet_name
import streamlit as st

st.title("Pets name generator") # open port 8051

animal_type = st.sidebar.selectbox("What is your pet?", ("Dog","Cat", "Cow","Goat"))

pet_color = st.sidebar.text_input(f"What color is your {animal_type}?", max_chars=20)

if pet_color:
    response = generate_pet_name(animal_type, pet_color)
    st.text(response)
