from graph.state import AgentState
from data_ingestion import retriever

def retrieve(state: AgentState) -> AgentState:
    print("--- RETRIEVE ---")

    # Operate on the question
    question = state["question"]
    documents = retriever.invoke(question)

    # Update the documents in the state and return the whole state
    state["documents"] = documents

    return state