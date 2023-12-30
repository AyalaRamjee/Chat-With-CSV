import streamlit as st 
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

@st.cache(allow_output_mutation=True)
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

st.title("Chat with CSV using Llama2 🦙🦜")
st.markdown("<h3 style='text-align: center; color: black;'>Built by Ramjee - You're Welcome</h3>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

import chardet
import pandas as pd
import io


if uploaded_file:
    st.info("CSV file uploaded.")
    
    # Detecting encoding of the file
    file_content = uploaded_file.getvalue()
    detected = chardet.detect(file_content)
    file_encoding = detected['encoding']
    st.info(f"Detected file encoding: {file_encoding}")
    
    # Read the CSV using pandas
    df = pd.read_csv(io.BytesIO(file_content), encoding=file_encoding)
    
    # Replace None or NaN values with empty strings
    df.fillna("", inplace=True)
    
    # Write the cleaned data to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding=file_encoding) as tmp_file:
        df.to_csv(tmp_file, index=False)
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding=file_encoding, csv_args={'delimiter': ','})
    
    # Extracting data using the detected encoding
    st.info("Extracting data from the file...")
    data = loader.load()
    st.success("Data extracted successfully!")
    
    # Generating embeddings
    st.info("Generating embeddings... This might take some time.")
    embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
    
    # Using the original method to generate embeddings
    db = FAISS.from_documents(data, embeddings_model)
    db.save_local(DB_FAISS_PATH)
    st.success("Embeddings generated successfully!")
    
    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
