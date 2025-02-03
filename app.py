import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with OPENAI"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries."),
        ("user", "Question:{question}"),
    ]
)


def generate_response(question, llm, temperature, max_tokens):
    llm = Ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})

    # Extract only the final answer (remove <think> reasoning part)
    if "<think>" in answer:
        # Split the response into <think> and final answer parts
        think_part, final_answer = answer.split("</think>", 1)
        return final_answer.strip()
    else:
        return answer


## Create Streamlit application
### Title of the app
st.title("Simple Q&A Chatbot with Open Source LLM's")

### Dropdown to select from various open source models
engine = st.sidebar.selectbox(
    "Select LLM model", ["deepseek-r1:1.5b", "gemma2:2b", "mistral"]
)

## Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, engine, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input")
