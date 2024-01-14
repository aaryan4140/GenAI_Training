## Integrate our code with Open AI API

import os
from constants import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory




import streamlit as st
os.environ["OPENAI_API_KEY"]= OPENAI_API_KEY


#streamlit framework

st.title("Celebrity info search")
input_text= st.text_input("search with celebrity name")

#MEMMORY

person_memmory= ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memmory=ConversationBufferMemory(input_key='person', memory_key='chat_history')
desc_memmory=ConversationBufferMemory(input_key='dob', memory_key='description_history')


#Prompt templating
    
first_input_prompt=PromptTemplate(

    input_variables=["name"],
    template="Tell me about {name} in 30 words"
)



#oprnai llm

llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm,prompt=first_input_prompt, verbose=True,output_key='person',memory=person_memmory)


second_input_prompt=PromptTemplate(

    input_variables=['person'],
    template="when was {person} born"
)

chain2=LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob',memory=dob_memmory)

third_input_prompt=PromptTemplate(

    input_variables=['dob'],
    template="mention 5 major event happen around {dob} in world"
)
chain3=LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description', memory=desc_memmory)



parent_chain = SequentialChain(
    chains=[chain,chain2,chain3], input_variables=['name'], output_variables=['person','dob','description'],verbose=True)



if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person_memmory.buffer)
    
    with st.expander('Major Event'):
        st.info(desc_memmory.buffer)




