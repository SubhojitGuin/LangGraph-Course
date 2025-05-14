from dotenv import load_dotenv

load_dotenv()

from pprint import pprint

from graph.chains.retrieval_grader import GradeDocument, retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader
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

def test_generation_chain() -> None:
    """Test the generation chain with a relevant document."""
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})
    pprint(generation)

def test_hallucination_grader_answer_yes() -> None:
    """Test the hallucination grader with a relevant document."""
    question = "agent memory"
    docs = retriever.invoke(question)
    
    generation = generation_chain.invoke({"question": question, "context": docs})
    res = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )

    assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
    """Test the hallucination grader with a relevant document."""
    question = "agent memory"
    docs = retriever.invoke(question)
    
    # generation = generation_chain.invoke({"question": question, "context": docs})
    res = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza, we need to first start with the dough",
        }
    )

    assert not res.binary_score