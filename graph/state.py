from typing import TypedDict, List

class AgentState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        question: question text
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]