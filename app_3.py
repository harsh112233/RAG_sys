import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_api_key

# App header and description
st.header("Context Chat with RAG")
st.write("Upload a document and ask questions about the data.")

# Initialize session state
if "chat_engine" not in st.session_state:
    st.session_state["chat_engine"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "documents" not in st.session_state:
    st.session_state.documents = None

# File upload
uploaded_file = st.file_uploader("**Upload a file**")
if uploaded_file is not None and st.session_state.documents is None:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            documents = SimpleDirectoryReader(temp_dir).load_data()
            st.session_state.documents = documents
            st.success(f"File '{uploaded_file.name}' uploaded and loaded successfully!")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Function to initialize chat engine
@st.cache_resource
def init_chat_engine():
    try:
        if st.session_state.documents is None:
            st.error("No documents loaded. Please upload a file first.")
            return None
        documents = st.session_state.documents
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=(
                "You are a chatbot, able to have normal interactions, as well as talk"
                " about an essay discussing Paul Graham's life."
            ),
        )
        return chat_engine
    except Exception as e:
        st.error(f"Failed to initialize chat engine: {e}")
        return None

# Load chat engine on button click
if st.button("Load Documents and Initialize Chat"):
    with st.spinner("Indexing documents..."):
        st.session_state["chat_engine"] = init_chat_engine()
        if st.session_state["chat_engine"]:
            st.success("Chat engine initialized successfully!")

# Chat interface
if st.session_state["chat_engine"]:
    st.subheader("Chat with the AI")
    user_query = st.text_input("Ask AI:", key="user_input")
    if user_query:
        if user_query.lower() == "exit":
            st.session_state["chat_history"].append({"role": "user", "content": "Exit"})
            st.session_state["chat_history"].append({"role": "assistant", "content": "Goodbye! Chat ended."})
            st.session_state["chat_engine"] = None
        else:
            try:
                with st.spinner("Generating response..."):
                    response = st.session_state["chat_engine"].chat(user_query)
                    st.session_state["chat_history"].append({"role": "user", "content": user_query})
                    st.session_state["chat_history"].append({"role": "assistant", "content": str(response)})
            except Exception as e:
                st.error(f"Error generating response: {e}")
    for message in st.session_state["chat_history"]:
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**AI:** {message['content']}")
else:
    st.warning("Please load documents and initialize the chat engine first.")

# Instructions
st.sidebar.markdown("""
### Instructions
1. Upload a text document using the file uploader.
2. Click 'Load Documents and Initialize Chat' to start.
3. Type your question and press Enter.
4. Type 'exit' to end the chat.
5. You can also delete and upload another file. It will reset the chat engine.
""")