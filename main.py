## Integrate our code with Open AI API

import os
from constants import OPENAI_API_KEY
from langchain.llms import OpenAI

import streamlit as st
os.environ["OPENAI_API_KEY"]= OPENAI_API_KEY


#streamlit framework

st.title("Langchain demo with OpenAPI")
input_text= st.text_input("search any topic want")

#oprnai llm

llm = OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))






