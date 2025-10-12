import os
from dotenv import load_dotenv
import yaml
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool, CalculatorTool
from crewai.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime, timedelta
from typing import Optional

# Load environment variables from .env file
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Load prompts from YAML file
with open("prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)

# Initialize the LLM with Gemini-2.5-flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key
)

# Initialize tools
web_search_tool = ScrapeWebsiteTool()
calculator_tool = CalculatorTool()

# Custom CalendarTool implementation
class CalendarTool(BaseTool):
    name: str = "Calendar Tool"
    description: str = "A tool to manage basic calendar operations like checking the current date/time, adding events, or listing upcoming events in a mock calendar."

    def _run(self, action: str, event_name: Optional[str] = None, days_ahead: Optional[int] = None) -> str:
        """
        Handles calendar actions:
        - 'current': Returns current date/time.
        - 'add': Adds an event (mock - returns confirmation with future date).
        - 'list': Lists mock upcoming events.
        """
        now = datetime.now()
        
        if action == "current":
            return f"Current date/time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        
        elif action == "add" and event_name and days_ahead is not None:
            future_date = now + timedelta(days=days_ahead)
            return f"Event '{event_name}' added successfully for {future_date.strftime('%Y-%m-%d')}."
        
        elif action == "list":
            mock_events = [
                "Meeting with team - 2025-10-15 10:00",
                "Project deadline - 2025-10-20 17:00"
            ]
            return "Upcoming events:\n" + "\n".join(mock_events)
        
        else:
            return "Invalid action. Use: 'current', 'add <event> <days_ahead>', or 'list'."

# Define the agent
general_agent = Agent(
    role="General Assistant",
    goal=prompts["system_prompts"]["general_agent"]["goal"],
    backstory=prompts["system_prompts"]["general_agent"]["backstory"],
    verbose=True,
    llm=llm,
    tools=[
        web_search_tool,
        calculator_tool,
        CalendarTool()
    ]
)

# Define a sample task
task = Task(
    description=prompts["task_prompts"]["sample_task"],
    agent=general_agent,
    expected_output="A concise response with the requested information or calculation."
)

# Create and run the crew
crew = Crew(
    agents=[general_agent],
    tasks=[task],
    verbose=True
)

# Execute the crew
result = crew.kickoff()

# Print the result
print(result)