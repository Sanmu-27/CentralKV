import json
import networkx as nx

class GraphExporter:
    """
    Exports the internal AttentionGraph to common formats (GEXF, JSON) 
    for analysis in Gephi or other tools.
    """
    def __init__(self, output_dir="graph_exports"):
        self.output_dir = output_dir
        
    def export_to_gexf(self, graph_data, step, filename=None):
        """
        Exports current graph state to GEXF format.
        
        Args:
            graph_data: (dict) Adjacency info or list of edges
            step: (int) Current generation step
        """
        G = nx.DiGraph()
        
        # Build graph from data (assuming adjacency list format)
        # nodes: {id: {label: "token", score: 0.5}}
        # edges: [(src, dst, weight)]
        
        nodes = graph_data.get("nodes", {})
        for nid, data in nodes.items():
            G.add_node(nid, label=data.get("token", str(nid)), score=data.get("score", 0.0))
            
        edges = graph_data.get("edges", [])
        for src, dst, weight in edges:
            G.add_edge(src, dst, weight=weight)
            
        fname = filename or f"{self.output_dir}/attention_graph_step_{step}.gexf"
        nx.write_gexf(G, fname)
        print(f"Exported graph to {fname}")
        
    def export_snapshot(self, cache_manager, step):
        # Implementation to extract state from KVCacheManager and export
        pass





