"""
A 3-agent CrewAI team: Researcher -> Compliance Reviewer -> Writer.
Sequential process - each agent's output becomes the next agent's input,
one after another, in a fixed order.
"""

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool as crewai_tool

from clinical_platform.agents.tools.clinical_tools import search_clinical_documents

# CrewAI wants its own @tool wrapper. We reuse the EXISTING search logic
# from Phase 8 rather than rebuilding it - just re-exposed for CrewAI.
@crewai_tool("Search Clinical Documents")
def crew_search_tool(query: str) -> str:
    """Search the clinical document library for information relevant to the query."""
    return search_clinical_documents.invoke({"query": query})


def build_crew(question: str, model_id: str = "gpt-4o-mini") -> Crew:
    researcher = Agent(
        role="Clinical Researcher",
        goal="Find accurate, relevant information from the clinical document library",
        backstory=(
            "You are a meticulous researcher who ONLY reports facts found in "
            "the document library. You never guess or use outside knowledge."
        ),
        tools=[crew_search_tool],
        llm=model_id,
        verbose=True,
    )

    reviewer = Agent(
        role="Compliance Reviewer",
        goal="Verify that findings are fully grounded in the source documents and actually answer the question",
        backstory=(
            "You are a strict compliance reviewer. You reject any finding "
            "that is not clearly supported by the retrieved document text, "
            "and flag if the finding doesn't actually address the question asked."
        ),
        llm=model_id,
        verbose=True,
    )

    writer = Agent(
        role="Clinical Writer",
        goal="Write a clear, accurate final answer for the end user based on the reviewed findings",
        backstory=(
            "You write clean, professional answers for healthcare "
            "professionals. You never add information beyond what the "
            "reviewer has approved."
        ),
        llm=model_id,
        verbose=True,
    )

    research_task = Task(
        description=f"Search the clinical documents to find information answering: {question}",
        expected_output="The relevant facts found, with their source document and section.",
        agent=researcher,
    )

    review_task = Task(
        description="Review the researcher's findings. Confirm they are grounded in the documents and directly answer the original question. Flag any gaps.",
        expected_output="An approval statement, or a list of concerns if the findings are insufficient.",
        agent=reviewer,
        context=[research_task],
    )

    writing_task = Task(
        description="Using the reviewed findings, write a clear, final answer to the original question for a healthcare professional.",
        expected_output="A clean, well-written final answer.",
        agent=writer,
        context=[review_task],
    )

    return Crew(
        agents=[researcher, reviewer, writer],
        tasks=[research_task, review_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )