import os.path

import streamlit as st

# from src.beam.server import BeamClient
# client = BeamClient('localhost:5000')
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from src.beam.llm import beam_llm

st.title('Beam Example')
st.write('This is a simple example of using Beam with Streamlit.')

# App title
# st.set_page_config(page_title="🤗💬 HugChat")

# Openai Credentials
with st.sidebar:
    st.title('OpenAI Chatbot 💬')

    api_key = st.text_input('Enter OpenAI api-key:', type='password')
    if not api_key:
        if 'llm' in st.session_state:
            st.session_state.pop('llm')
        st.warning('Please enter your credentials!', icon='⚠️')
    else:
        if 'llm' not in st.session_state:
            st.session_state['llm'] = beam_llm('openai:///gpt-4', api_key=api_key)
        st.success('Proceed to entering your prompt message!', icon='👉')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
# def generate_response(prompt_input, email, passwd):
#     # Hugging Face Login
#     sign = Login(email, passwd)
#     cookies = sign.login()
#     # Create ChatBot
#     chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
#     return chatbot.chat(prompt_input)


# User-provided prompt
if prompt := st.chat_input(disabled=False):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = st.session_state['llm']
            response = llm.chat(prompt).text
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

