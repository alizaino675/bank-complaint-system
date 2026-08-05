from rag_tools import rag_tool
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent 
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY') or os.getenv('API_KEY')
tools = [rag_tool()]

llama = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    api_key=api_key
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional bank support agent. Always use the search_bank_policies tool to search for bank rules before answering customer complaints."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llama, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

