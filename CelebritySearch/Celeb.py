## Integrate our code with Open AI API

import os
from constants import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain




import streamlit as st
os.environ["OPENAI_API_KEY"]= OPENAI_API_KEY


#streamlit framework

st.title("Celebrity info search")
input_text= st.text_input("search with celebrity name")


#Prompt templating
    
first_input_prompt=PromptTemplate(

    input_variables=["name"],
    template="Tell me about {name} in 30 words"
)

#oprnai llm

llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm,prompt=first_input_prompt, verbose=True,output_key='person')


second_input_prompt=PromptTemplate(

    input_variables=['person'],
    template="when was {person} born"
)

chain2=LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob')

parent_chain = SequentialChain(
    chains=[chain,chain2], input_variables=['name'], output_variables=['person','dob'],verbose=True)



if input_text:
    st.write(parent_chain({'name':input_text}))




