from typing import Dict, Any
from langchain.schema import Document
from langchain_tavily import TavilySearch
from graph.state import AgentState

from dotenv import load_dotenv

load_dotenv(override=True)

web_search_tool = TavilySearch(max_results=3)

def web_search(state: AgentState) -> Dict[str, Any]:
    print("---WEB SEARCH---")

    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )

    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})