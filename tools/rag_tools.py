from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('GEMINI_API_KEY', '')


index_path = "C:/Users/user/Desktop/Codes/patients/faiss_index"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectore_store = FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True)

retriver = vectore_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})


def rag_tool():
    
    
    tool = create_retriever_tool(
        retriever=retriver,
        name = "search_bank_policies",
        description= "Use this tool to access official bank policies.It should be available whenever there are any questions regarding payment terms credit card procedures, loan repayments, or financial disputes."
    )
    return tool

tools = [rag_tool()]
model = ChatGoogleGenerativeAI(
    model='gemini-3.6-flash',
    temperature=0,
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a professional bank support agent. Always use the search_bank_policies tool to search for bank rules before answering customer complaints."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

def get_ansewr(user_input: str):
    
    agent  =  create_tool_calling_agent(llm=model, tools=tools, prompt=prompt)
    agent_excuter = AgentExecutor(agent=agent, tools=tools)
    resposne = agent_excuter.invoke({"input": user_input})
    return resposne['output']





