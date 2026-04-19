from langgraph.graph import StateGraph, END, START
from agent.nodes import (
    IncidentState,
    analyze_incident_node,
    react_agent_node,
    tool_node,
    final_node
)

def should_continue(state: IncidentState):
    action = state.get("action")

    if action == "finish":
        return "finish"
    elif action == "search":
        return "search"
    else:
        raise ValueError(f"Unknown action: {action}")

def build_graph():
    graph = StateGraph(IncidentState)
    graph.add_node("analyze", analyze_incident_node)
    graph.add_node("agent", react_agent_node)
    graph.add_node("search", tool_node)
    graph.add_node("finish", final_node)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "search": "search",
            "finish": "finish"
        }
    )
    graph.add_edge("search", "agent")
    graph.add_edge("finish", END)
    return graph.compile()
incident_graph = build_graph()