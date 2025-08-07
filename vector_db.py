import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def pdf_database(question):
    file_path = "./"
    pdf_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.pdf')]

    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()

        # ✅ Combine all page content into one big document
        full_text = " ".join([page.page_content for page in pages])

        # ✅ Now split into semantically meaningful chunks
        chunks = splitter.create_documents([full_text])
        all_chunks.extend(chunks)

    # ✅ Build vector store from properly chunked documents
    vector_store = InMemoryVectorStore.from_documents(all_chunks, OpenAIEmbeddings())

    # ✅ Run semantic search
    documents = vector_store.similarity_search(question, k=2)

    return "\n---\n".join([doc.page_content for doc in documents])
