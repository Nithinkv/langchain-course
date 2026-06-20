from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient
from schemas import AgentResponse

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

def main():

    result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": query,
        }
    ]
    })

    print(result)

if __name__ == "__main__":
    main()
