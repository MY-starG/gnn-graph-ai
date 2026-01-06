"""
Graph Augmentation Utilities for GNN Training

This module provides modern graph augmentation techniques:
1. Edge dropping - Randomly remove edges
2. Node dropping - Randomly remove nodes and their connections
3. Feature masking - Randomly mask node features
4. Subgraph sampling - Extract random subgraphs
5. Edge perturbation - Add random edges
6. Attribute noise - Add Gaussian noise to features

These techniques improve model robustness and generalization
for research publication and PhD applications
"""

import torch
import numpy as np
from torch_geometric.utils import dropout_edge, to_undirected, add_self_loops, remove_self_loops
from torch_geometric.data import Data
import copy


class GraphAugmentor:
    """Comprehensive graph augmentation toolkit"""
    
    def __init__(self, augmentation_config=None):
        """
        Initialize graph augmentor
        
        Args:
            augmentation_config: Dictionary with augmentation parameters
                {
                    'edge_drop_rate': 0.1,
                    'node_drop_rate': 0.1,
                    'feature_mask_rate': 0.1,
                    'edge_add_rate': 0.1,
                    'feature_noise_std': 0.1
                }
        """
        self.config = augmentation_config or {
            'edge_drop_rate': 0.0,
            'node_drop_rate': 0.0,
            'feature_mask_rate': 0.0,
            'edge_add_rate': 0.0,
            'feature_noise_std': 0.0
        }
        
        print("Graph Augmentor initialized")
        print(f"Configuration: {self.config}")
    
    def edge_dropout(self, edge_index, drop_rate=None):
        """
        Randomly drop edges from the graph
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            drop_rate: Probability of dropping an edge
        
        Returns:
            Augmented edge index
        """
        if drop_rate is None:
            drop_rate = self.config.get('edge_drop_rate', 0.0)
        
        if drop_rate == 0.0:
            return edge_index
        
        edge_index_aug, _ = dropout_edge(
            edge_index,
            p=drop_rate,
            force_undirected=True,
            training=True
        )
        
        return edge_index_aug
    
    def node_dropout(self, data, drop_rate=None):
        """
        Randomly drop nodes and their associated edges
        
        Args:
            data: PyG Data object
            drop_rate: Probability of dropping a node
        
        Returns:
            Augmented Data object
        """
        if drop_rate is None:
            drop_rate = self.config.get('node_drop_rate', 0.0)
        
        if drop_rate == 0.0:
            return data
        
        num_nodes = data.x.size(0)
        
        # Create node mask (keep nodes with probability 1 - drop_rate)
        node_mask = torch.rand(num_nodes) > drop_rate
        
        # Filter nodes
        x_aug = data.x[node_mask]
        
        # Create mapping from old indices to new indices
        old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
        old_to_new[node_mask] = torch.arange(node_mask.sum())
        
        # Filter and remap edges
        edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
        edge_index_aug = data.edge_index[:, edge_mask]
        edge_index_aug = old_to_new[edge_index_aug]
        
        # Create new data object
        data_aug = Data(x=x_aug, edge_index=edge_index_aug)
        
        # Copy other attributes if they exist
        for key in data.keys():
            if key not in ['x', 'edge_index'] and hasattr(data, key):
                attr = getattr(data, key)
                if isinstance(attr, torch.Tensor) and attr.size(0) == num_nodes:
                    setattr(data_aug, key, attr[node_mask])
                else:
                    setattr(data_aug, key, attr)
        
        return data_aug
    
    def feature_masking(self, x, mask_rate=None):
        """
        Randomly mask node features
        
        Args:
            x: Node feature matrix [num_nodes, num_features]
            mask_rate: Probability of masking a feature
        
        Returns:
            Augmented feature matrix
        """
        if mask_rate is None:
            mask_rate = self.config.get('feature_mask_rate', 0.0)
        
        if mask_rate == 0.0:
            return x
        
        x_aug = x.clone()
        
        # Create mask
        mask = torch.rand_like(x) < mask_rate
        
        # Set masked features to zero
        x_aug[mask] = 0
        
        return x_aug
    
    def feature_noise(self, x, noise_std=None):
        """
        Add Gaussian noise to node features
        
        Args:
            x: Node feature matrix [num_nodes, num_features]
            noise_std: Standard deviation of Gaussian noise
        
        Returns:
            Augmented feature matrix
        """
        if noise_std is None:
            noise_std = self.config.get('feature_noise_std', 0.0)
        
        if noise_std == 0.0:
            return x
        
        noise = torch.randn_like(x) * noise_std
        x_aug = x + noise
        
        return x_aug
    
    def edge_perturbation(self, edge_index, num_nodes, add_rate=None):
        """
        Add random edges to the graph
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            num_nodes: Total number of nodes
            add_rate: Rate of edges to add (relative to existing edges)
        
        Returns:
            Augmented edge index
        """
        if add_rate is None:
            add_rate = self.config.get('edge_add_rate', 0.0)
        
        if add_rate == 0.0:
            return edge_index
        
        num_edges = edge_index.size(1)
        num_add = int(num_edges * add_rate)
        
        # Generate random edges
        new_edges = torch.randint(0, num_nodes, (2, num_add), dtype=torch.long)
        
        # Concatenate with existing edges
        edge_index_aug = torch.cat([edge_index, new_edges], dim=1)
        
        # Remove self-loops and make undirected
        edge_index_aug = remove_self_loops(edge_index_aug)[0]
        edge_index_aug = to_undirected(edge_index_aug)
        
        return edge_index_aug
    
    def subgraph_sampling(self, data, sample_rate=0.7):
        """
        Sample a random subgraph
        
        Args:
            data: PyG Data object
            sample_rate: Fraction of nodes to sample
        
        Returns:
            Augmented Data object
        """
        num_nodes = data.x.size(0)
        num_sample = int(num_nodes * sample_rate)
        
        # Randomly sample nodes
        sampled_nodes = torch.randperm(num_nodes)[:num_sample]
        sampled_nodes = torch.sort(sampled_nodes)[0]
        
        # Create node mask
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        node_mask[sampled_nodes] = True
        
        # Filter features
        x_aug = data.x[node_mask]
        
        # Create mapping
        old_to_new = torch.full((num_nodes,), -1, dtype=torch.long)
        old_to_new[node_mask] = torch.arange(node_mask.sum())
        
        # Filter and remap edges
        edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
        edge_index_aug = data.edge_index[:, edge_mask]
        edge_index_aug = old_to_new[edge_index_aug]
        
        # Create new data object
        data_aug = Data(x=x_aug, edge_index=edge_index_aug)
        
        return data_aug
    
    def augment(self, data, augmentation_types=None):
        """
        Apply multiple augmentations sequentially
        
        Args:
            data: PyG Data object
            augmentation_types: List of augmentation types to apply
                                ['edge_drop', 'node_drop', 'feature_mask', 'feature_noise', 'edge_perturb']
        
        Returns:
            Augmented Data object
        """
        if augmentation_types is None:
            augmentation_types = []
            if self.config.get('edge_drop_rate', 0.0) > 0:
                augmentation_types.append('edge_drop')
            if self.config.get('node_drop_rate', 0.0) > 0:
                augmentation_types.append('node_drop')
            if self.config.get('feature_mask_rate', 0.0) > 0:
                augmentation_types.append('feature_mask')
            if self.config.get('feature_noise_std', 0.0) > 0:
                augmentation_types.append('feature_noise')
            if self.config.get('edge_add_rate', 0.0) > 0:
                augmentation_types.append('edge_perturb')
        
        data_aug = copy.deepcopy(data)
        
        for aug_type in augmentation_types:
            if aug_type == 'edge_drop':
                data_aug.edge_index = self.edge_dropout(data_aug.edge_index)
            
            elif aug_type == 'node_drop':
                data_aug = self.node_dropout(data_aug)
            
            elif aug_type == 'feature_mask':
                data_aug.x = self.feature_masking(data_aug.x)
            
            elif aug_type == 'feature_noise':
                data_aug.x = self.feature_noise(data_aug.x)
            
            elif aug_type == 'edge_perturb':
                data_aug.edge_index = self.edge_perturbation(
                    data_aug.edge_index,
                    data_aug.x.size(0)
                )
        
        return data_aug
    
    def create_multiple_views(self, data, num_views=2):
        """
        Create multiple augmented views of the same graph
        Useful for contrastive learning
        
        Args:
            data: PyG Data object
            num_views: Number of augmented views to create
        
        Returns:
            List of augmented Data objects
        """
        views = []
        for _ in range(num_views):
            view = self.augment(data)
            views.append(view)
        
        return views


def compare_augmentation_effects(augmentation_configs, augmentation_names=None):
    """
    Helper function to compare different augmentation strategies
    
    Args:
        augmentation_configs: List of augmentation config dictionaries
        augmentation_names: Optional list of names for each configuration
    
    Returns:
        Dictionary mapping names to GraphAugmentor objects
    """
    if augmentation_names is None:
        augmentation_names = [f"Config_{i}" for i in range(len(augmentation_configs))]
    
    augmentors = {}
    for name, config in zip(augmentation_names, augmentation_configs):
        augmentors[name] = GraphAugmentor(config)
    
    return augmentors


# ===================== PREDEFINED AUGMENTATION STRATEGIES =====================

AUGMENTATION_PRESETS = {
    'light': {
        'edge_drop_rate': 0.1,
        'node_drop_rate': 0.0,
        'feature_mask_rate': 0.1,
        'edge_add_rate': 0.0,
        'feature_noise_std': 0.0
    },
    
    'moderate': {
        'edge_drop_rate': 0.2,
        'node_drop_rate': 0.1,
        'feature_mask_rate': 0.2,
        'edge_add_rate': 0.1,
        'feature_noise_std': 0.05
    },
    
    'aggressive': {
        'edge_drop_rate': 0.3,
        'node_drop_rate': 0.2,
        'feature_mask_rate': 0.3,
        'edge_add_rate': 0.2,
        'feature_noise_std': 0.1
    },
    
    'edge_only': {
        'edge_drop_rate': 0.3,
        'node_drop_rate': 0.0,
        'feature_mask_rate': 0.0,
        'edge_add_rate': 0.0,
        'feature_noise_std': 0.0
    },
    
    'feature_only': {
        'edge_drop_rate': 0.0,
        'node_drop_rate': 0.0,
        'feature_mask_rate': 0.3,
        'edge_add_rate': 0.0,
        'feature_noise_std': 0.1
    },
    
    'node_edge': {
        'edge_drop_rate': 0.2,
        'node_drop_rate': 0.15,
        'feature_mask_rate': 0.0,
        'edge_add_rate': 0.1,
        'feature_noise_std': 0.0
    }
}


def get_preset_augmentor(preset_name):
    """
    Get a predefined augmentor configuration
    
    Args:
        preset_name: Name of the preset ('light', 'moderate', 'aggressive', etc.)
    
    Returns:
        GraphAugmentor object with preset configuration
    """
    if preset_name not in AUGMENTATION_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(AUGMENTATION_PRESETS.keys())}")
    
    config = AUGMENTATION_PRESETS[preset_name]
    return GraphAugmentor(config)


# ===================== EXAMPLE USAGE =====================

def demonstrate_augmentations():
    """Demonstrate different augmentation techniques"""
    from torch_geometric.datasets import Planetoid
    
    print("="*80)
    print("GRAPH AUGMENTATION DEMONSTRATION")
    print("="*80)
    print()
    
    # Load sample data
    print("Loading Cora dataset...")
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    
    print(f"Original graph: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Feature dimensions: {data.x.size(1)}")
    print()
    
    # Test different augmentations
    augmentation_tests = [
        ('No Augmentation', {}),
        ('Edge Dropout (20%)', {'edge_drop_rate': 0.2}),
        ('Node Dropout (10%)', {'node_drop_rate': 0.1}),
        ('Feature Masking (20%)', {'feature_mask_rate': 0.2}),
        ('Feature Noise (std=0.1)', {'feature_noise_std': 0.1}),
        ('Edge Perturbation (10%)', {'edge_add_rate': 0.1}),
        ('Combined (Light)', AUGMENTATION_PRESETS['light']),
        ('Combined (Moderate)', AUGMENTATION_PRESETS['moderate']),
        ('Combined (Aggressive)', AUGMENTATION_PRESETS['aggressive'])
    ]
    
    print("Testing augmentations:")
    print("-"*80)
    
    for name, config in augmentation_tests:
        augmentor = GraphAugmentor(config)
        data_aug = augmentor.augment(data)
        
        edge_reduction = (1 - data_aug.num_edges / data.num_edges) * 100
        node_reduction = (1 - data_aug.num_nodes / data.num_nodes) * 100
        feature_diff = torch.abs(data_aug.x - data.x[:data_aug.num_nodes]).mean().item()
        
        print(f"\n{name}:")
        print(f"  Nodes: {data_aug.num_nodes} ({node_reduction:.1f}% reduction)")
        print(f"  Edges: {data_aug.num_edges} ({edge_reduction:.1f}% change)")
        print(f"  Feature diff: {feature_diff:.4f}")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print()


if __name__ == '__main__':
    print(__doc__)
    print("\nAvailable augmentation presets:")
    for preset_name in AUGMENTATION_PRESETS.keys():
        print(f"  - {preset_name}")
    print("\nRun demonstrate_augmentations() to see examples")
    print()
    
    # Uncomment to run demonstration
    # demonstrate_augmentations()