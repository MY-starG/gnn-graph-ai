import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


class CoraDataLoader:
    def __init__(self, root='data/Cora'):
        transform = NormalizeFeatures()
        self.dataset = Planetoid(root=root, name='Cora', transform=transform)
        self.data = self.dataset[0]
        
    def get_data(self):
        return self.data
    
    def get_num_features(self):
        return self.dataset.num_features
    
    def get_num_classes(self):
        return self.dataset.num_classes


if __name__ == "__main__":
    loader = CoraDataLoader()
    data = loader.get_data()
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Features: {loader.get_num_features()}")
    print(f"Classes: {loader.get_num_classes()}")
    print(f"Train nodes: {data.train_mask.sum()}")