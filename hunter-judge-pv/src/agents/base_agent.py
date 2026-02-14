from abc import ABC, abstractmethod
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from src.core.schemas import AgentState

from dotenv import load_dotenv
load_dotenv()

class BasePharmacovigilanceAgent(ABC):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.workflow = StateGraph(AgentState)
        self.app = None

    @abstractmethod
    def extract_entities_node(self, state: AgentState) -> AgentState:
        """Extract entities from the input text."""
        raise NotImplementedError

    @abstractmethod
    def verify_drug_reaction_node(self, state: AgentState) -> AgentState:
        """Verify the extracted drug-reaction pair against knowledge base."""
        raise NotImplementedError

    @abstractmethod
    def assess_causality_node(self, state: AgentState) -> AgentState:
        """Assess causality using Naranjo scale."""
        raise NotImplementedError

    @abstractmethod
    def reflect_node(self, state: AgentState) -> dict:
        """Reflect on failures and optimize search terms."""
        raise NotImplementedError

    def check_verification_status(self, state: AgentState) -> str:
        """
        Conditional Edge: Decides whether to reflect or proceed to assessment.
        """
        print("---EDGE: CHECKING VERIFICATION STATUS---")
        evidence = state.get("verification_evidence", "")
        retries = state.get("retries", 0)

        # If verification failed and we haven't exceeded retries, go to reflect node.
        if "no relevant information found" in evidence.lower() and retries < 2:
            print("Verification failed. Routing to reflect.")
            return "reflect"
        else:
            print("Verification succeeded or max retries reached. Routing to assess causality.")
            return "assess_causality"

    def decide_next_step(self, state: AgentState) -> str:
        """
        Edge: Decides flow based on state and retry limits.
        """
        entities = state.get("entities")
        # If we have entities, verify them. 
        if entities and entities.drug and entities.reaction:
            return "verify" 

        return "end"

    def compile(self):
        """Define the graph structure and compile it."""
        # --- NODES ---
        self.workflow.add_node("extract_entities", self.extract_entities_node)
        self.workflow.add_node("verify_drug_reaction", self.verify_drug_reaction_node)
        self.workflow.add_node("reflect", self.reflect_node)
        self.workflow.add_node("assess_causality", self.assess_causality_node)

        # --- EDGES ---
        self.workflow.set_entry_point("extract_entities")

        self.workflow.add_conditional_edges(
            "extract_entities",
            self.decide_next_step,
            {
                "verify": "verify_drug_reaction",
                "end": END
            }
        )
        
        self.workflow.add_conditional_edges(
            "verify_drug_reaction",
            self.check_verification_status,
            {"assess_causality": "assess_causality", "reflect": "reflect"},
        )
        self.workflow.add_edge("reflect", "verify_drug_reaction")
        self.workflow.add_edge("assess_causality", END)

        self.app = self.workflow.compile()

    def run(self, text: str) -> dict:
        """Runs the agentic workflow on a given piece of text."""
        # Child classes or the specific implementation of run logic can override this if needed,
        # but the basic invocation is shared.
        initial_state = AgentState(
            text=text, 
            entities=None,
            normalized_text=None,
            current_drug_query=None,
            current_reaction_query=None,
            verification_evidence=None, 
            assessment=None, 
            retries=0,
            error=None
        )
        
        final_state = self.app.invoke(initial_state, {"recursion_limit": 10})
        
        return {
            "assessment": final_state.get("assessment"),
            "evidence": final_state.get("verification_evidence"),
            "used_search_term": final_state.get("current_drug_query"),
            "current_reaction_query": final_state.get("current_reaction_query"),
            "retries": final_state.get("retries", 0)
        }
