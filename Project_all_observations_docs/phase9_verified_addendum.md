## Phase 9 — CrewAI Multi-Agent Team (LIVE VERIFIED)

**What it does:** 3 specialist agents work together sequentially - 
Researcher (searches documents) -> Compliance Reviewer (checks 
grounding/accuracy) -> Writer (produces final clean answer). Different 
from Phase 8's single agent, which is ONE AI juggling everything itself.

**Live proof, actually seen in the Streamlit UI:**

Question: "What vitamin deficiency is linked to long-term metformin use?"
-> Researcher found the relevant excerpt from drug_manual_metformin.md
-> Compliance Reviewer confirmed the finding was grounded and complete
-> Writer produced: "Long-term use of metformin has been associated with 
decreased absorption of vitamin B12, which can lead to vitamin B12 
deficiency. It is important for healthcare professionals to be aware 
of this potential side effect and to monitor patients..."

All 3 agent outputs visible individually in expandable sections in the 
UI, plus the final combined answer - genuinely watched the handoff 
happen, not just a single black-box response.

**Real dependency conflict resolved along the way (good interview 
material):** Adding CrewAI created a real version conflict - my project 
needed openai>=3.2.0 for langchain-openai, but CrewAI's ecosystem was 
built against an older `openai>=1.13.3` range. I researched CrewAI's 
actual documented requirements, found the numbers weren't truly 
incompatible (just needed the right minimum, no artificial upper cap), 
and resolved it by letting `uv`'s dependency resolver do its job instead 
of guessing version pins myself. Verified afterward that my existing 
Phase 4 OpenAI provider code still worked correctly after the SDK 
version change - it did, because I'd used stable, version-independent 
API syntax.

**One sentence to remember:** *"I built a 3-agent CrewAI team where each 
agent has ONE narrow job - I chose sequential handoff over a dynamic 
manager pattern because the workflow (research -> verify -> write) has 
a naturally fixed order, and added complexity from a manager agent 
wasn't justified here."*

**Comparing Phase 8 (LangGraph) vs Phase 9 (CrewAI) - a real interview 
talking point:** *"LangGraph gave me full control over the exact loop 
logic and made it easy to inject a fake LLM for free, instant testing. 
CrewAI's role-based abstraction was faster to set up for a fixed, 
sequential multi-specialist workflow, but harder to fully fake for 
testing - so I tested its STRUCTURE for free (are the right agents/
tasks/tools wired correctly) and ran the actual AI execution sparingly, 
as a deliberate, costed confirmation step, not a routine test."*

**Status: ✅ FULLY DONE - structure tests pass for free, AND live 
end-to-end execution confirmed visually through the Streamlit UI with 
a correct, accurate final answer.**
