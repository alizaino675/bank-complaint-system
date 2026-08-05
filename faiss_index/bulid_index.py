from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


files_path = "C:/Users/user/Desktop/Codes/patients/knowledg_base"

index_path = "C:/Users/user/Desktop/Codes/patients/faiss_index"
def built_index():
    loader = PyPDFDirectoryLoader(
        files_path
    )
    docs = loader.load()

    text_siplliter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=500
    )

    docs = text_siplliter.split_documents(documents=docs)

    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    vecctor_store = FAISS.from_documents(documents=docs, embedding=embeddings)

    vecctor_store.save_local(index_path)

if __name__ == "__main__":
    built_index()