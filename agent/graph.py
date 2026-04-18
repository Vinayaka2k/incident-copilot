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
    if action == "final":
        return "final"
    return "tool"


def build_graph():
    graph = StateGraph(IncidentState)
    graph.add_node("analyze", analyze_incident_node)
    graph.add_node("agent", react_agent_node)
    graph.add_node("tool", tool_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tool": "tool",
            "final": "final"
        }
    )
    graph.add_edge("tool", "agent")
    graph.add_edge("final", END)
    return graph.compile()
incident_graph = build_graph()















