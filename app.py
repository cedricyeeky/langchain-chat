import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import  FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore


def get_conversational_chain(vectorstore):


    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3, convert_system_message_to_human=True)

    # prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    # chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    # return chain
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



def user_input(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    # new_db = FAISS.load_local("faiss_index", embeddings)
    # docs = new_db.similarity_search(user_question)

    # chain = get_conversational_chain(vectorstore)

    
    # response = chain(
    #     {"input_documents":docs, "question": user_question}
    #     , return_only_outputs=True)

    # print(response)
    # st.write("Reply: ", response["output_text"])

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Reverse chat history in order to display latest query at the top
    reversed_chat_history = reversed(st.session_state.chat_history)

    if reversed_chat_history:  # Check if the list is not empty
        for i, message in enumerate(reversed_chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
                
    return st.session_state.chat_history




def main():

    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    st.write(css, unsafe_allow_html=True)

    user_question = st.text_input("Ask a Question from the PDF Files")

    # if "conversation" not in st.session_state:
    #     st.session_state.conversation = None
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = None

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button. You may upload multiple files!", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                # create vector store
                vectorstore = get_vector_store(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversational_chain(vectorstore)
                st.success("Done")



if __name__ == "__main__":
    main()