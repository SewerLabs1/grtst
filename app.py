import os
import gradio as gr
from langchain.chat_models.fireworks import ChatFireworks
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory

os.environ["FIREWORKS_API_KEY"] = "ku9UYtzjSAATlcAstO8yrB89MzvDqJL3lGIkNgnVZ7URxPxK"

llm = ChatFireworks(
    model="accounts/fireworks/models/mistral-7b-instruct-4k",
    model_kwargs={"temperature": 0.2, "max_tokens": 445, "top_p": 0.9},
)

def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format)
    return gpt_response.content

gr.themes.Monochrome()

gr.ChatInterface(predict).launch(share=True)