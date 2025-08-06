from pprint import pprint

from dotenv import load_dotenv

load_dotenv()

from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from data_ingestion import retriever
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations

def test_retrieval_grader_answers_yes() -> None:

    question = "agent memory"

    doc = retriever.invoke(input=question)
    doc_text = doc[0].page_content

    response : GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )

    assert response.binary_score.upper() == "YES"


def test_retrieval_grader_answers_no() -> None:

    question = "agent memory"

    doc = retriever.invoke(input=question)
    doc_text = doc[0].page_content

    response : GradeDocuments = retrieval_grader.invoke(
        {"question": "Where is Dhanesh working?", "document": doc_text}
    )

    assert response.binary_score.upper() == "NO"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(input=question)
    generation = generation_chain.invoke({"context": docs, "question": question})

    pprint(generation)


def test_hallucination_grader_answers_yes() -> None:

    question = "agent memory"
    docs = retriever.invoke(input=question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    response: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )

    assert response.binary_score


def test_hallucination_grader_answers_no() -> None:

    question = "agent memory"
    docs = retriever.invoke(input=question)

    response: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": "Dhanesh works for OCBC Bank"}
    )

    assert not response.binary_score