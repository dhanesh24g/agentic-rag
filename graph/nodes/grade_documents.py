from graph.state import AgentState
from graph.chains.retrieval_grader import retrieval_grader
from typing import Dict, Any

def grade_documents(state: AgentState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_documents = []
    web_search = False

    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        if score.binary_score.upper() == "YES":
            print("---GRADE: DOCUMENT IS RELEVANT---")
            filtered_documents.append(doc)
        else:
            web_search = True
            print("---GRADE: DOCUMENT IS NOT RELEVANT---")
            continue

    return {"documents": filtered_documents, "question": question, "web_search": web_search}