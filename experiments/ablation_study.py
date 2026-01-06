"""
Comprehensive Ablation Study for GNN Models

This script systematically tests the effect of different architectural components:
1. Number of layers (depth)
2. Hidden dimensions (width)
3. Dropout rates
4. GAT attention heads
5. Activation functions
6. Aggregation methods (for GraphSAGE)

Designed to demonstrate deep understanding of model architectures
for research publication and PhD applications
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
import json
from itertools import product

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


# ===================== CONFIGURABLE MODEL ARCHITECTURES =====================

class ConfigurableGCN(torch.nn.Module):
    """GCN with configurable architecture"""
    def __init__(self, num_features, hidden_channels, num_classes, 
                 num_layers=2, dropout=0.5, activation='relu'):
        super(ConfigurableGCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        
        self.convs = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(num_features, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, num_classes))
    
    def get_activation(self):
        if self.activation == 'relu':
            return F.relu
        elif self.activation == 'elu':
            return F.elu
        elif self.activation == 'leaky_relu':
            return F.leaky_relu
        else:
            return F.relu
    
    def forward(self, x, edge_index):
        act_fn = self.get_activation()
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = act_fn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class ConfigurableGAT(torch.nn.Module):
    """GAT with configurable architecture"""
    def __init__(self, num_features, hidden_channels, num_classes,
                 heads=8, num_layers=2, dropout=0.6):
        super(ConfigurableGAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(num_features, hidden_channels, heads=heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                     heads=heads, dropout=dropout))
        
        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, num_classes, 
                                 heads=1, concat=False, dropout=dropout))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class ConfigurableGraphSAGE(torch.nn.Module):
    """GraphSAGE with configurable architecture"""
    def __init__(self, num_features, hidden_channels, num_classes,
                 num_layers=3, dropout=0.5, aggr='mean'):
        super(ConfigurableGraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(num_features, hidden_channels, aggr=aggr))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, num_classes, aggr=aggr))
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


# ===================== TRAINING AND EVALUATION =====================

def train_model(model, data, optimizer, epochs=200, early_stopping=50, verbose=False):
    """Train model with early stopping"""
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    
    return train_acc, val_acc, test_acc


def run_single_experiment(model_class, data, config, device='cpu'):
    """Run single experiment with given configuration"""
    model = model_class(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    train_acc, val_acc, test_acc = train_model(model, data, optimizer, epochs=200, early_stopping=30)
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'config': config
    }


# ===================== ABLATION STUDIES =====================

class AblationStudyManager:
    """Manages comprehensive ablation studies"""
    
    def __init__(self, data, output_dir='./results/ablation_studies'):
        self.data = data
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {
            'gcn': [],
            'gat': [],
            'graphsage': []
        }
        
        print(f"Ablation Study Manager initialized")
        print(f"Output directory: {output_dir}")
    
    def study_num_layers(self, model_type='gcn', layer_range=[2, 3, 4, 5]):
        """Study effect of number of layers"""
        print(f"\nStudying number of layers for {model_type.upper()}...")
        print(f"Testing layers: {layer_range}")
        
        base_config = {
            'num_features': self.data.num_features,
            'hidden_channels': 64,
            'num_classes': self.data.y.max().item() + 1,
            'dropout': 0.5
        }
        
        model_classes = {
            'gcn': ConfigurableGCN,
            'gat': ConfigurableGAT,
            'graphsage': ConfigurableGraphSAGE
        }
        
        if model_type == 'gat':
            base_config['heads'] = 8
        
        results = []
        for num_layers in layer_range:
            print(f"  Testing {num_layers} layers...")
            config = base_config.copy()
            config['num_layers'] = num_layers
            
            result = run_single_experiment(model_classes[model_type], self.data, config)
            result['variable'] = 'num_layers'
            result['value'] = num_layers
            results.append(result)
            
            print(f"    Train: {result['train_acc']:.4f}, Val: {result['val_acc']:.4f}, Test: {result['test_acc']:.4f}")
        
        self.results[model_type].extend(results)
        return results
    
    def study_hidden_dimensions(self, model_type='gcn', hidden_dims=[32, 64, 128, 256]):
        """Study effect of hidden dimensions"""
        print(f"\nStudying hidden dimensions for {model_type.upper()}...")
        print(f"Testing dimensions: {hidden_dims}")
        
        base_config = {
            'num_features': self.data.num_features,
            'num_classes': self.data.y.max().item() + 1,
            'num_layers': 3,
            'dropout': 0.5
        }
        
        model_classes = {
            'gcn': ConfigurableGCN,
            'gat': ConfigurableGAT,
            'graphsage': ConfigurableGraphSAGE
        }
        
        if model_type == 'gat':
            base_config['heads'] = 8
        
        results = []
        for hidden_dim in hidden_dims:
            print(f"  Testing hidden_dim={hidden_dim}...")
            config = base_config.copy()
            config['hidden_channels'] = hidden_dim
            
            result = run_single_experiment(model_classes[model_type], self.data, config)
            result['variable'] = 'hidden_channels'
            result['value'] = hidden_dim
            results.append(result)
            
            print(f"    Train: {result['train_acc']:.4f}, Val: {result['val_acc']:.4f}, Test: {result['test_acc']:.4f}")
        
        self.results[model_type].extend(results)
        return results
    
    def study_dropout_rates(self, model_type='gcn', dropout_rates=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8]):
        """Study effect of dropout rates"""
        print(f"\nStudying dropout rates for {model_type.upper()}...")
        print(f"Testing rates: {dropout_rates}")
        
        base_config = {
            'num_features': self.data.num_features,
            'hidden_channels': 64,
            'num_classes': self.data.y.max().item() + 1,
            'num_layers': 3
        }
        
        model_classes = {
            'gcn': ConfigurableGCN,
            'gat': ConfigurableGAT,
            'graphsage': ConfigurableGraphSAGE
        }
        
        if model_type == 'gat':
            base_config['heads'] = 8
        
        results = []
        for dropout in dropout_rates:
            print(f"  Testing dropout={dropout}...")
            config = base_config.copy()
            config['dropout'] = dropout
            
            result = run_single_experiment(model_classes[model_type], self.data, config)
            result['variable'] = 'dropout'
            result['value'] = dropout
            results.append(result)
            
            print(f"    Train: {result['train_acc']:.4f}, Val: {result['val_acc']:.4f}, Test: {result['test_acc']:.4f}")
        
        self.results[model_type].extend(results)
        return results
    
    def study_attention_heads(self, head_counts=[1, 2, 4, 8, 16]):
        """Study effect of attention heads for GAT"""
        print(f"\nStudying attention heads for GAT...")
        print(f"Testing head counts: {head_counts}")
        
        base_config = {
            'num_features': self.data.num_features,
            'hidden_channels': 64,
            'num_classes': self.data.y.max().item() + 1,
            'num_layers': 2,
            'dropout': 0.6
        }
        
        results = []
        for heads in head_counts:
            print(f"  Testing {heads} heads...")
            config = base_config.copy()
            config['heads'] = heads
            
            result = run_single_experiment(ConfigurableGAT, self.data, config)
            result['variable'] = 'heads'
            result['value'] = heads
            results.append(result)
            
            print(f"    Train: {result['train_acc']:.4f}, Val: {result['val_acc']:.4f}, Test: {result['test_acc']:.4f}")
        
        self.results['gat'].extend(results)
        return results
    
    def study_activation_functions(self, activations=['relu', 'elu', 'leaky_relu']):
        """Study effect of activation functions for GCN"""
        print(f"\nStudying activation functions for GCN...")
        print(f"Testing activations: {activations}")
        
        base_config = {
            'num_features': self.data.num_features,
            'hidden_channels': 64,
            'num_classes': self.data.y.max().item() + 1,
            'num_layers': 3,
            'dropout': 0.5
        }
        
        results = []
        for activation in activations:
            print(f"  Testing {activation}...")
            config = base_config.copy()
            config['activation'] = activation
            
            result = run_single_experiment(ConfigurableGCN, self.data, config)
            result['variable'] = 'activation'
            result['value'] = activation
            results.append(result)
            
            print(f"    Train: {result['train_acc']:.4f}, Val: {result['val_acc']:.4f}, Test: {result['test_acc']:.4f}")
        
        self.results['gcn'].extend(results)
        return results
    
    def study_aggregation_methods(self, aggr_methods=['mean', 'max', 'add']):
        """Study effect of aggregation methods for GraphSAGE"""
        print(f"\nStudying aggregation methods for GraphSAGE...")
        print(f"Testing methods: {aggr_methods}")
        
        base_config = {
            'num_features': self.data.num_features,
            'hidden_channels': 64,
            'num_classes': self.data.y.max().item() + 1,
            'num_layers': 3,
            'dropout': 0.5
        }
        
        results = []
        for aggr in aggr_methods:
            print(f"  Testing {aggr} aggregation...")
            config = base_config.copy()
            config['aggr'] = aggr
            
            result = run_single_experiment(ConfigurableGraphSAGE, self.data, config)
            result['variable'] = 'aggregation'
            result['value'] = aggr
            results.append(result)
            
            print(f"    Train: {result['train_acc']:.4f}, Val: {result['val_acc']:.4f}, Test: {result['test_acc']:.4f}")
        
        self.results['graphsage'].extend(results)
        return results
    
    def visualize_results(self, model_type, variable):
        """Create visualization for a specific ablation study"""
        model_results = [r for r in self.results[model_type] if r['variable'] == variable]
        
        if not model_results:
            print(f"No results found for {model_type} - {variable}")
            return
        
        # Sort by value
        if isinstance(model_results[0]['value'], str):
            # Categorical variable
            values = [r['value'] for r in model_results]
            indices = range(len(values))
        else:
            # Numerical variable
            model_results = sorted(model_results, key=lambda x: x['value'])
            values = [r['value'] for r in model_results]
            indices = values
        
        train_accs = [r['train_acc'] for r in model_results]
        val_accs = [r['val_acc'] for r in model_results]
        test_accs = [r['test_acc'] for r in model_results]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Line plot
        ax = axes[0]
        if isinstance(values[0], (int, float)):
            ax.plot(values, train_accs, 'o-', linewidth=2, markersize=8, label='Train', alpha=0.8)
            ax.plot(values, val_accs, 's-', linewidth=2, markersize=8, label='Validation', alpha=0.8)
            ax.plot(values, test_accs, '^-', linewidth=2, markersize=8, label='Test', alpha=0.8)
            ax.set_xlabel(variable.replace('_', ' ').title(), fontweight='bold')
        else:
            ax.plot(indices, train_accs, 'o-', linewidth=2, markersize=8, label='Train', alpha=0.8)
            ax.plot(indices, val_accs, 's-', linewidth=2, markersize=8, label='Validation', alpha=0.8)
            ax.plot(indices, test_accs, '^-', linewidth=2, markersize=8, label='Test', alpha=0.8)
            ax.set_xticks(indices)
            ax.set_xticklabels(values, rotation=45 if len(str(values[0])) > 5 else 0)
            ax.set_xlabel(variable.replace('_', ' ').title(), fontweight='bold')
        
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title(f'{model_type.upper()} - Effect of {variable.replace("_", " ").title()}', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Bar plot for test accuracy
        ax = axes[1]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(test_accs)))
        bars = ax.bar(range(len(test_accs)), test_accs, color=colors, alpha=0.8, edgecolor='black')
        
        if isinstance(values[0], str):
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values, rotation=45 if len(str(values[0])) > 5 else 0)
        else:
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values, rotation=45 if len(str(values[0])) > 5 else 0)
        
        ax.set_ylabel('Test Accuracy', fontweight='bold')
        ax.set_xlabel(variable.replace('_', ' ').title(), fontweight='bold')
        ax.set_title('Test Accuracy Comparison', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Add value labels
        for bar, acc in zip(bars, test_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, f'{model_type}_{variable}_ablation.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {save_path}")
    
    def create_summary_table(self, model_type):
        """Create summary table for all ablation studies"""
        model_results = self.results[model_type]
        
        if not model_results:
            print(f"No results for {model_type}")
            return
        
        # Group by variable
        variables = set(r['variable'] for r in model_results)
        
        summary_data = []
        for variable in variables:
            var_results = [r for r in model_results if r['variable'] == variable]
            
            # Find best configuration
            best_result = max(var_results, key=lambda x: x['test_acc'])
            
            summary_data.append({
                'Variable': variable,
                'Best Value': best_result['value'],
                'Test Accuracy': best_result['test_acc'],
                'Val Accuracy': best_result['val_acc'],
                'Train Accuracy': best_result['train_acc']
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, f'{model_type}_ablation_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Summary table saved to {csv_path}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best test accuracy
        test_acc_col = list(df.columns).index('Test Accuracy')
        best_idx = df['Test Accuracy'].idxmax()
        table[(best_idx + 1, test_acc_col)].set_facecolor('#FFD700')
        
        plt.title(f'{model_type.upper()} - Ablation Study Summary', 
                 fontsize=14, fontweight='bold', pad=20)
        
        table_path = os.path.join(self.output_dir, f'{model_type}_ablation_table.png')
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary table visualization saved to {table_path}")
        
        return df
    
    def save_all_results(self):
        """Save all results to JSON"""
        json_path = os.path.join(self.output_dir, 'all_ablation_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"All results saved to {json_path}")


# ===================== MAIN EXECUTION =====================

def main():
    print("="*90)
    print("COMPREHENSIVE ABLATION STUDIES FOR GNN MODELS")
    print("="*90)
    print()
    
    # Load data
    print("Loading Cora dataset...")
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges\n")
    
    # Initialize ablation study manager
    manager = AblationStudyManager(data)
    
    # GCN Ablation Studies
    print("\n" + "="*90)
    print("GCN ABLATION STUDIES")
    print("="*90)
    manager.study_num_layers('gcn', layer_range=[2, 3, 4, 5])
    manager.study_hidden_dimensions('gcn', hidden_dims=[32, 64, 128, 256])
    manager.study_dropout_rates('gcn', dropout_rates=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8])
    manager.study_activation_functions(activations=['relu', 'elu', 'leaky_relu'])
    
    # Visualize GCN results
    print("\nGenerating GCN visualizations...")
    manager.visualize_results('gcn', 'num_layers')
    manager.visualize_results('gcn', 'hidden_channels')
    manager.visualize_results('gcn', 'dropout')
    manager.visualize_results('gcn', 'activation')
    manager.create_summary_table('gcn')
    
    # GAT Ablation Studies
    print("\n" + "="*90)
    print("GAT ABLATION STUDIES")
    print("="*90)
    manager.study_num_layers('gat', layer_range=[2, 3, 4])
    manager.study_hidden_dimensions('gat', hidden_dims=[32, 64, 128])
    manager.study_dropout_rates('gat', dropout_rates=[0.0, 0.3, 0.6, 0.8])
    manager.study_attention_heads(head_counts=[1, 2, 4, 8, 16])
    
    # Visualize GAT results
    print("\nGenerating GAT visualizations...")
    manager.visualize_results('gat', 'num_layers')
    manager.visualize_results('gat', 'hidden_channels')
    manager.visualize_results('gat', 'dropout')
    manager.visualize_results('gat', 'heads')
    manager.create_summary_table('gat')
    
    # GraphSAGE Ablation Studies
    print("\n" + "="*90)
    print("GRAPHSAGE ABLATION STUDIES")
    print("="*90)
    manager.study_num_layers('graphsage', layer_range=[2, 3, 4, 5])
    manager.study_hidden_dimensions('graphsage', hidden_dims=[32, 64, 128, 256])
    manager.study_dropout_rates('graphsage', dropout_rates=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8])
    manager.study_aggregation_methods(aggr_methods=['mean', 'max', 'add'])
    
    # Visualize GraphSAGE results
    print("\nGenerating GraphSAGE visualizations...")
    manager.visualize_results('graphsage', 'num_layers')
    manager.visualize_results('graphsage', 'hidden_channels')
    manager.visualize_results('graphsage', 'dropout')
    manager.visualize_results('graphsage', 'aggregation')
    manager.create_summary_table('graphsage')
    
    # Save all results
    manager.save_all_results()
    
    print("\n" + "="*90)
    print("ABLATION STUDIES COMPLETE")
    print("="*90)
    print(f"\nAll results saved to: {manager.output_dir}")
    print("\nGenerated outputs:")
    print("  - Ablation plots for each variable (PNG)")
    print("  - Summary tables (CSV + PNG)")
    print("  - Complete results (JSON)")
    print("\nDemonstrates deep understanding of architectural components!")
    print()


if __name__ == '__main__':
    main()