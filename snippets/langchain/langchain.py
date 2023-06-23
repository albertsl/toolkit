from langchain.llms import OpenAI
llm = OpenAI(openai_api_key="", temperature=0.9) # Temperature of 0.9, generates responses that are more random. A low number would give us highly accurate responses.
llm.predict("Write a story about a mountain next to the beach.")

# Using Chat Models:
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
chat.predict_messages([HumanMessage(content="Translate this sentence from English to French. I love programming.")])