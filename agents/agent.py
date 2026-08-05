from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from tools.rag_tools import get_ansewr
from crewai import LLM
from dotenv import load_dotenv
import os

llm_model = LLM(
    model='gemini/gemini-3.6-flash',
    temperature=0,
    api_key=os.getenv('GEMINI_API_KEY', '')
)


@tool('Query Bank Policy')
def query_search(complaint_text: str):
    """Queries the bank policy database for relevant information based on the complaint text."""
    if complaint_text:
        try:
            return get_ansewr(user_input= complaint_text)
        except Exception as e:
            return f'Error quering policies: {str(e)}.'


def run_complaint_crew(complaint_text: str, category: str, action_team:str):
    #creating agent for research for information that revelant to the docs
    resolution_specialist = Agent(
        role="Senior Bank Resolution Specialist",
        goal=f"Search bank policies using the tool and formulate a clear solution assigned to {action_team}",
        backstory=(
            f"You represent the {action_team}. Your role is to query official bank policies "
            "and draft a direct, polite resolution for the customer."
        ),
        memory=False,
        tools=[query_search],
        llm=llm_model,
        )

    
    #initiate the task for the research agent
    resolution_task = Task(
        description=(
            f"Analyze the customer complaint: '{complaint_text}'. "
            f"First, use the 'Query Bank Policy' tool to retrieve bank rules. "
            f"Then, write a complete and professional response on behalf of {action_team}."
        ),
        expected_output="A complete, polite, and professional resolution message.",
        agent=resolution_specialist,
    )

    crew = Crew(
        agents=[resolution_specialist],
        tasks=[resolution_task],
        process= Process.sequential
    )

    result = crew.kickoff()
    return str(result)





    

