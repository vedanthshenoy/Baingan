import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Annotated

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Agentic RAG Chat App", description="Simple agentic app using LangGraph with web search and weather tools")

# Load Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

# Initialize the LLM (Gemini-2.5-flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# Define the system prompt
system_prompt = """
You are a helpful assistant with access to two tools: web search and weather lookup.
Use the web search tool when the user asks for current events, facts, or information not in your knowledge.
Use the weather tool when the user asks about weather in a specific city.
If the query doesn't require tools, respond directly.
Always be concise and helpful.
"""

# Custom weather tool (uses free wttr.in API, no key needed)
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    try:
        response = requests.get(f"http://wttr.in/{city}?format=3")
        if response.status_code == 200:
            return f"Weather in {city}: {response.text.strip()}"
        else:
            return f"Could not fetch weather for {city}. Please try again."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# Web search tool (free DuckDuckGo)
web_search = DuckDuckGoSearchRun()

# List of tools
tools = [web_search, get_weather]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Define the agent state using TypedDict
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | ToolMessage], add_messages]

# Define the agent node
def agent_node(state: AgentState) -> Dict[str, List[Any]]:
    messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = prompt | llm_with_tools
    result = chain.invoke({"messages": messages})
    return {"messages": [result]}

# Define the tool node
def tool_node(state: AgentState) -> Dict[str, List[Any]]:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return state
    
    results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        for tool in tools:
            if tool.name == tool_name:
                result = tool.invoke(tool_args)
                results.append(ToolMessage(
                    content=str(result),
                    name=tool_name,
                    tool_call_id=tool_call["id"]
                ))
    return {"messages": results}

# Define the router to decide next step
def router(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Create the LangGraph workflow
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("agent", router, {"tools": "tools", END: END})
workflow.set_entry_point("agent")
agent = workflow.compile()

# Pydantic model for request body
class ChatRequest(BaseModel):
    query: str

# API endpoint for chat
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Invoke the agent with initial state
        result = agent.invoke({"messages": [HumanMessage(content=request.query)]})
        
        # Extract the last AI message content
        response_content = result["messages"][-1].content if result["messages"][-1].type == "ai" else "No response generated."
        
        return {"response": response_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Agentic Chat App is running! Use POST /chat with {'query': 'your question'}."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)