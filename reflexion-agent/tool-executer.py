from langchain_tavily import TavilySearch
from langchain_core.tools import (
    StructuredTool,
)  # Convert functions to tools by passing structured data to the LLM aiding in its use
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, ReviseAnswer
from typing import List
from dotenv import load_dotenv

load_dotenv()

tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: List[str], **kwargs):
    return tavily_tool.batch({"query": query} for query in search_queries)


execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)
