import os
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# Import Base Class and Schemas
from base_agent import BasePharmacovigilanceAgent
from src.core.schemas import AgentState

# --- 1. OUTPUT SCHEMA (Structured Verdict) ---
class VerificationResult(BaseModel):
    is_ade: bool = Field(description="True if this is a side effect, False if Indication/Negation/Safe")
    reasoning: str = Field(description="Brief explanation (e.g., 'Classified as Indication', 'Explicit Negation')")

# --- 2. PROMPT (The "Context Judge") ---
CADEC_VERIFICATION_PROMPT = """You are a Senior Pharmacovigilance Safety Officer.
Your task is to determine if the following text describes a 
**Side Effect (Adverse Drug Event)** caused by the drug: "{drug_context}".

**INPUT:**
- Text: {text}
- Drug Context: {drug_context}

**RULES FOR CLASSIFICATION:**
1. **YES (True ADE):**
   - The text describes a physiological issue (pain, dysfunction, mental state).
   - The symptom occurred *after* or *during* the use of the drug.
   - Example: "Lipitor gave me a headache." -> YES

2. **NO (False Positive):**
   - **Indication:** The symptom is the *reason* they took the drug.
     - Example: "I took Lipitor for high cholesterol." -> NO
   - **Negation:** The user explicitly says the symptom is absent.
     - Example: "I had no side effects." or "No headache." -> NO
   - **Pre-existing:** The symptom existed before taking the drug.
     - Example: "I have always had back pain." -> NO
   - **Cost/Logistics:** Complaints about price or packaging. -> NO

**OUTPUT:**
Return a JSON object with:
- `is_ade`: boolean
- `reasoning`: string (Be concise)
"""

class CADECAgent(BasePharmacovigilanceAgent):
    def __init__(self):
        super().__init__()
        self.compile()

    # --- NODE IMPLEMENTATION ---
    def verify_drug_reaction_node(self, state: AgentState) -> dict:
        """
        The main Logic Node. It acts as the 'Judge'.
        """
        text = state["text"]
        drug_context = state.get("current_drug_query", "Unknown Drug")

        parser = PydanticOutputParser(pydantic_object=VerificationResult)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", CADEC_VERIFICATION_PROMPT),
            ("human", "Analyze this case.")
        ])

        chain = prompt | self.llm | parser

        try:
            result = chain.invoke({
                "text": text,
                "drug_context": drug_context
            })
            
            # Map the result back to the AgentState structure
            # We treat 'is_ade' as the assessment result
            return {
                **state, 
                "assessment": result, # Store the Pydantic object here
                "error": None
            }
        except Exception as e:
            print(f"   [Agent Error] {e}")
            return {**state, "assessment": None, "error": str(e)}

    # --- REQUIRED ABSTRACT METHODS (Implemented as pass/dummies) ---
    def extract_entities_node(self, state: AgentState) -> AgentState:
        pass # Not used in this simplified pipeline

    def assess_causality_node(self, state: AgentState) -> AgentState:
        pass

    def reflect_node(self, state: AgentState) -> dict:
        pass

    # --- OVERRIDE GRAPH STRUCTURE ---
    def compile(self):
        """
        Builds a linear graph: START -> VERIFY -> END
        We bypass the complex retry logic of the Base Class.
        """
        self.workflow.add_node("verify", self.verify_drug_reaction_node)
        self.workflow.set_entry_point("verify")
        self.workflow.add_edge("verify", END)
        self.app = self.workflow.compile()

    # --- OVERRIDE RUN ---
    def run(self, text: str, drug_context: str) -> dict:
        """
        Custom run method that accepts 'drug_context'.
        """
        initial_state = AgentState(
            text=text,
            current_drug_query=drug_context, # Storing context here
            entities=None,
            normalized_text=None,
            current_reaction_query=None,
            verification_evidence=None,
            assessment=None,
            retries=0,
            error=None
        )
        
        final_state = self.app.invoke(initial_state)
        
        # Extract the Pydantic result
        assessment = final_state.get("assessment")
        
        if assessment:
            return {
                "is_ade": assessment.is_ade,
                "reasoning": assessment.reasoning
            }
        else:
            # Fallback if LLM failed
            return {"is_ade": False, "reasoning": "Agent Error"}