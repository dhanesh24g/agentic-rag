from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.constants import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEB_SEARCH
from graph.nodes import retrieve, grade_documents, generate, web_search
from graph.state import AgentState

load_dotenv()

def decide_to_generate(state: AgentState):
    print("---ASSESS THE GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: SOME OF THE DOCUMENTS ARE NOT RELEVANT TO QUESTION---"
        )
        return WEB_SEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_doc_and_question(state: AgentState) -> str:
    print("---CHECK HALLUCINATIONS---")

    question = state["question"]
    docs = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({
        "documents": docs,
        "generation": generation
    })

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION VS QUESTION---")

        score = answer_grader.invoke(
            {"question": question, "generation": generation},
        )

        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES THE QUESTION---")
            return "relevant"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS THE QUESTION---")
            return "not relevant"

    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "hallucinated"

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE
    }
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_doc_and_question,
    {
        "hallucinated": GENERATE,
        "not relevant": WEB_SEARCH,
        "relevant": END,
    }
)

workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="./graph.png")