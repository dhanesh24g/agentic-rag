from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0
)

class GradeDocuments(BaseModel):
    """Binary score for checking the relevance of the retrieved documents"""

    binary_score: str = Field(
        description="Whether the Documents are relevant to the question, 'YES' or 'NO'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_message = """You are a grader, assessing the relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'YES' or 'NO' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader