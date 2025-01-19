import json
import os
import pandas as pd
import boto3
import streamlit as st
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# AWS Bedrock client setup
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

## Data ingestion
def data_ingestion(file_list):
    """Process uploaded Excel files into LangChain Documents."""
    all_docs = []

    for file in file_list:
        df = pd.read_excel(file)
        docs = [
            Document(
                page_content=row.to_string(),
                metadata={"file_name": file.name, "row_index": index}
            )
            for index, row in df.iterrows()
        ]
        all_docs.extend(docs)

    return all_docs

## Vector Embedding and vector store
def get_vector_store(docs):
    """Create and save a FAISS vector store."""
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_llama3_llm():
    """Create the Anthropic Llama3 model."""
    llm = Bedrock(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len': 512})
    return llm

# Prompt template for QA
prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but summarize with 
at least 250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    """Retrieve answers using the LLM and vector store."""
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config(page_title="Chat with Uploaded Excel Files", layout="wide")
    
    st.header("Chat with Excel Files using AWS Bedrock 💁")

    user_question = st.text_input("Ask a Question from the Uploaded Excel Files")

    uploaded_files = st.file_uploader(
        "Upload one or more Excel files",
        type=["xlsx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.sidebar:
            st.title("Update or Create Vector Store:")
            
            if st.button("Process Uploaded Files"):
                with st.spinner("Processing uploaded files..."):
                    docs = data_ingestion(uploaded_files)
                    get_vector_store(docs)
                    st.success("Vector store updated successfully!")

        if st.button("Get Llama3 Output"):
            with st.spinner("Processing your query..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                llm = get_llama3_llm()
                answer = get_response_llm(llm, faiss_index, user_question)
                st.write(answer)
                st.success("Done!")

if __name__ == "__main__":
    main()
