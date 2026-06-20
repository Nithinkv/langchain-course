from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient
from schemas import AgentResponse
from pydantic import BaseModel, Field

tavily = TavilyClient()

query = "climate in Tokya Today"
def Search(query:str):
    """Tool that search for a query using over the internet for the date 20-06-2026
    Args:
        query (str): The query to search for
        
    Returns:
        list: List of search results
    """
    print("searching for", query)
    return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-4o")
tools = [Search]


agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)

class Source(BaseModel):
    """schema for a source used by the agent"""
    url:str = Field(description="url of the source")

class AgentResponse(BaseModel):
    """schema for the response of the agent"""
    answer:str = Field(description="answer to the query")
    sources: list[Source] = Field(default_factory=list,description="list of sources used to answer the query")
    
def main():

    result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": query,
        }
    ]
    })

    print(result["structured_response"].sources)
    
if __name__ == "__main__":
    main()
