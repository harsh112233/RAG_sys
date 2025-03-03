## Retrieval augmented generation

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer

chat_engine = None

def test_chat_engine():
    documents=SimpleDirectoryReader("data").load_data()
    index=VectorStoreIndex.from_documents(documents, show_progress=True)
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=(
            "You are a chatbot, able to have normal interactions, as well as talk"
            " about an essay discussing Paul Grahams life."
        ),
    )

    return chat_engine

if chat_engine == None:
    chat_engine = test_chat_engine()

def chat_message(user_query):
    response = chat_engine.chat(user_query)
    return response

# user_query = input("Enter your query: ")
# response = chat_engine.chat(user_query)
# print(response)

# Continuous chat loop
while True:
    user_query = input("Enter your query (or type 'Exit' to quit): ")
    
    # Exit the loop if the user types "Exit"
    if user_query.lower() == "exit":
        print("Exiting the chat. Goodbye!")
        break
    
    # Get the response from the chat engine
    response = chat_message(user_query)
    print(response)