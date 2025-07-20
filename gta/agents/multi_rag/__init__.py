"""
Multi-VectorStore RAG Agent.

This agent supports two approaches:
1. Legacy: Single parallel search node (create_multi_rag_graph)
2. New: Individual search nodes for each vectorstore (create_multi_rag_graph_with_individual_nodes)
"""

from gta.agents.multi_rag.graph import (
    create_multi_rag_graph, 
    create_multi_rag_graph_with_individual_nodes,
    create_conditional_multi_rag_example,
    multi_rag_graph,
    legacy_graph
)
from gta.agents.multi_rag.state import MultiRAGState

__all__ = [
    "create_multi_rag_graph", 
    "create_multi_rag_graph_with_individual_nodes",
    "create_conditional_multi_rag_example",
    "multi_rag_graph", 
    "legacy_graph",
    "MultiRAGState"
] 