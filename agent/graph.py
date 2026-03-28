from langgraph.graph import START, END, StateGraph
from agent.nodes import (
    IncidentState, analyze_incident_node,
    rewrite_query_node, incident_search_node, triage_planning_node
)

def build_graph():
    """
    Build and compile the IncidentCopilot LangGraph
    FLow:
    Start -> analyze_incident -> rewrite_query ->   incident_search -> triage_planning -> END
    """
    graph = StateGraph(IncidentState)
    graph.add_node("analyze_incident", analyze_incident_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("incident_search", incident_search_node)
    graph.add_node("triage_planning", triage_planning_node)

    # Add linear edges
    graph.add_edge(START, "analyze_incident")
    graph.add_edge("analyze_incident", "rewrite_query")
    graph.add_edge("rewrite_query", "incident_search")
    graph.add_edge("incident_search", "triage_planning")
    graph.add_edge("triage_planning", END)
    
    # Compile graph
    return graph.compile()




















