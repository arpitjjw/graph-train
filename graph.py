import networkx as nx
from node2vec import Node2Vec
from gensim.models import KeyedVectors


class graph:
    def __init__(self, node_data=None, edge_data=None):
        """Initialize a graph object using dataframes from skills database."""
        self.node_df = node_data
        self.edge_df = edge_data
        self.G=nx.DiGraph()
        self.node_df.apply(lambda x: self.G.add_node(x['skill_id']) ,axis=1)
        self.edge_df.apply(lambda x: self.G.add_edge(x['subject_id'],x['object_id'],relation=x['relation']) ,axis=1)


    def add_node(self, node):
        self.G.add_node(node)
    
    def remove_node(self, node):
        self.G.remove_node(node)

    def add_edge(self, edge):  # edge is tuple or list of tuples
        self.G.add_edge(edge)

    def node2vec(self,dimensions,walk_length,num_walks,workers,q):
        node2vec = Node2Vec(self.G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers,q=q)
        self.node2vec = node2vec.fit(window=10, min_count=1, batch_words=4)
        self.wv=self.node2vec.wv

    def load_node2vec(self,path):
        self.wv=KeyedVectors.load(path)
    def save_node2vec(self, path):
        self.wv.save(path)  # path with .kv extension



    