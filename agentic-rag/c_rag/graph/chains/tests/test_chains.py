from dotenv import load_dotenv

load_dotenv()

from graph.chains.retrieval_grader import GradeDocument, retrieval_grader
from ingestion import retriever


def test_retreival_grader_answer_yes() -> None:
    """Test the retrieval grader with a relevant document."""
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[0].page_content

    res: GradeDocument = retrieval_grader.invoke(
        {"document": doc_text, "question": question}
    )

    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    """Test the retrieval grader with a non-relevant document."""
    question = "how to make pizza"
    docs = retriever.invoke(question)
    doc_text = docs[1].page_content

    res: GradeDocument = retrieval_grader.invoke(
        {"document": doc_text, "question": question}
    )

    assert res.binary_score == "no"