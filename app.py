from langchain_community.llms import Ollama 
import streamlit as st

model = "/scratch/alcobaaj/interpares/llama3-main/Meta-Llama-3-8B-Instruct" #"llama3:70b"
llm = Ollama(model=model)

st.title("Chatbot using Llama3")

prompt = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            st.write_stream(llm.stream(prompt, stop=['<|eot_id|>']))