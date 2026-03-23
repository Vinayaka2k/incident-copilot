from langgraph.graph import END,START,StateGraph
from agent.nodes import (
    IncidentState, analyze_incident_node, rewrite_query_node, 
    incident_search_node, triage_planning_node
)
def build_triage_graph():
    """
    Build and compile the strict linear 4 node LangGraph workflow
    Flow:
     Start -> analyze_incident
     -> rewrite query
     -> incident search
     -> triage planning
     -> END
    """
    graph_builder = StateGraph(IncidentState)
    graph_builder.add_node("analyze incident" , analyze_incident_node)
    graph_builder.add_node("rewrite query", rewrite_query_node)
    graph_builder.add_node("incident search", incident_search_node)
    graph_builder.add_node("traige planning", triage_planning_node)

    graph_builder.add_edge(START, "analyze incident")
    graph_builder.add_edge("analyze incident", "rewrite query")
    graph_builder.add_edge("rewrite query", "incident search")
    graph_builder.add_edge("incident search", "triage planning")
    graph_builder.add_edge("triage planning", END)

triage_graph = build_triage_graph()

def run_triage_agent(incident: str) -> IncidentState:
    """
    Run the compiled Langgaraph triage workflow fora single incident
    ArgS:
    Incident: Raw incident description from the user
    Returns: Final incident state afterall 4 nodes are complte
    """
    if not incident or not incident.strip():
        raise ValueError("incident cant be emptry")
    initial_state: IncidentState = {
        "incident": incident.strip()
    }
    final_state = triage_graph.invoke(initial_state)
    return final_state