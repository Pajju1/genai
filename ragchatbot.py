import streamlit as st
import pdfplumber 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv

load_dotenv()

st.header("Ask your files anything!")
st.markdown("### Upload your PDF file")
file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set! Please set it as an environment variable.")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chain" not in st.session_state:
    st.session_state.chain = None

if file is not None:
    # Process only if not already processed
    if st.session_state.vector_store is None:
        with st.spinner("Processing PDF..."):
            #Extract text from it
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"

            #split text into chunk
            text_splitter = RecursiveCharacterTextSplitter(
                separators = ["\n\n", "\n", ". ", " ", ""],
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(text)

            #generating embeddings
            embeddings = OpenAIEmbeddings(
                model = "text-embedding-3-small",
                openai_api_key = OPENAI_API_KEY
            )

            #store embeddings in vector db
            vector_store = FAISS.from_texts(chunks, embeddings)
            
            #get user question
            #user_question = st.text_input("Type your question here", key="input_question")

            #generate answer
            def format_docs(docs):
                return "\n\n".join([doc.page_content for doc in docs])
    
            #retrive from search
            retriver = vector_store.as_retriever(
                search_type = "mmr",
                search_kwargs={"k":4}
            )

            #Define the LLm and prompts
            llm = ChatOpenAI(
                model = "gpt-4o-mini",
                temperature = 0.3,
                max_tokens=1000,
                openai_api_key=OPENAI_API_KEY
            )

            #provide the prompts
            prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are a helpful assistant answering questions about a PDF document. \n\n"
            "Guidelines: \n"
            "1. Provide complete, well-explained answers using the context below. \n"
            "2. Include relevant details, numbers, and explanations to give a thorough response. \n"
            "3. If the context mentions related information, include it to give fuller picture. \n"
            "4. Only use information from the provided context - do not use outside knowledge. \n"
            "5. Summarize long information, ideally in bullets where needed \n"
            "6. If the information is not in the context, say so politely.\n"
            "Context: \n{context}"), 
            ("human", "{question}")
            ])

            chain = (
                {"context": retriver | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # Save to session state
            st.session_state.vector_store = vector_store
            st.session_state.chain = chain

        st.success("File processed successfully!")

    # Question input - outside the processing block with unique key
    user_question = st.text_input("Type your question here", key="user_question")

    if user_question:
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(user_question)
            st.write(response)

else:
    st.info("👆 Please upload a PDF file to begin.")

if __name__ == "__main__":
    pass