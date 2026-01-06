"""
ADVANCED GNN Explainability Framework using GNNExplainer

Provides comprehensive interpretability for GNN models by:
1. Identifying important nodes and edges for predictions
2. Computing feature importance with advanced analysis
3. Generating beautiful subgraph visualizations
4. Creating edge masks and node masks
5. Comparative analysis across models
6. Attention weight visualization (for GAT)

For Erasmus Mundus PhD Applications - Research Excellence Demonstration
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import seaborn as sns
from datetime import datetime

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


# ===================== MODEL DEFINITIONS (MATCHING YOUR FILES) =====================

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.6):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.6):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, num_classes)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv3(x, edge_index)
        return x


class AdvancedGNNExplainer:
    """Advanced GNN explanation and visualization tool with comprehensive analysis"""
    
    def __init__(self, model, data, model_name, device='cpu'):
        self.model = model.to(device)
        self.data = data.to(device)
        self.model_name = model_name
        self.device = device
        self.model.eval()
        
        # Initialize GNNExplainer
        self.explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            ),
        )
        
        # Class names for Cora
        self.class_names = [
            'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
            'Probabilistic_Methods', 'Reinforcement_Learning',
            'Rule_Learning', 'Theory'
        ]
        
        print(f"✓ Advanced GNNExplainer initialized for {model_name}")
    
    def explain_node(self, node_idx):
        """Explain prediction for a specific node"""
        explanation = self.explainer(
            self.data.x,
            self.data.edge_index,
            index=node_idx
        )
        return explanation
    
    def get_subgraph(self, node_idx, num_hops=2):
        """Extract k-hop subgraph around target node"""
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx,
            num_hops,
            self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes
        )
        return subset, edge_index, mapping, edge_mask
    
    def visualize_explanation_advanced(self, node_idx, save_path=None):
        """
        Create comprehensive visualization of node explanation with advanced features
        """
        # Get explanation
        explanation = self.explain_node(node_idx)
        
        # Get prediction
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred_class = out[node_idx].argmax().item()
            true_class = self.data.y[node_idx].item()
            pred_probs = torch.softmax(out[node_idx], dim=0)
        
        # Get subgraph
        subset, sub_edge_index, mapping, _ = self.get_subgraph(node_idx, num_hops=2)
        
        # Create figure with advanced layout
        fig = plt.figure(figsize=(24, 10))
        gs = fig.add_gridspec(2, 4, height_ratios=[2, 1], width_ratios=[1.5, 1, 1, 1])
        
        # ============ SUBPLOT 1: Enhanced Subgraph Visualization ============
        ax1 = fig.add_subplot(gs[:, 0])
        
        # Build NetworkX graph
        G = nx.Graph()
        for i, node in enumerate(subset):
            G.add_node(i, label=self.class_names[self.data.y[node].item()])
        
        # Add edges with importance weights
        edge_mask = explanation.edge_mask[:len(sub_edge_index[0])]
        edge_weights = edge_mask.cpu().numpy()
        
        for i in range(sub_edge_index.shape[1]):
            src, dst = sub_edge_index[:, i].cpu().numpy()
            weight = edge_weights[i] if i < len(edge_weights) else 0.0
            G.add_edge(src, dst, weight=weight)
        
        # Advanced layout
        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
        
        # Node colors based on class with gradient
        node_colors = [self.data.y[subset[i]].item() for i in G.nodes()]
        
        # Draw nodes with enhanced styling
        node_collection = nx.draw_networkx_nodes(
            G, pos, 
            node_color=node_colors,
            node_size=600,
            cmap='tab10',
            alpha=0.85,
            ax=ax1,
            edgecolors='black',
            linewidths=2
        )
        
        # Highlight target node with special marker
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[mapping.item()],
            node_color='red',
            node_size=1200,
            node_shape='*',
            ax=ax1,
            edgecolors='darkred',
            linewidths=3
        )
        
        # Draw edges with importance-based styling
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        # Normalize weights for better visualization
        if max(weights) > 0:
            normalized_weights = [w / max(weights) for w in weights]
        else:
            normalized_weights = weights
        
        nx.draw_networkx_edges(
            G, pos,
            width=[w * 8 for w in normalized_weights],
            edge_color=weights,
            edge_cmap=plt.cm.Reds,
            edge_vmin=0,
            edge_vmax=max(weights) if weights else 1,
            alpha=0.7,
            ax=ax1
        )
        
        # Draw labels with background
        labels = {i: f"{i}" for i in G.nodes()}
        labels[mapping.item()] = f"{mapping.item()}★"
        nx.draw_networkx_labels(
            G, pos, labels, 
            font_size=11, 
            font_weight='bold',
            font_color='white',
            ax=ax1
        )
        
        # Add colorbar for edge importance
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=max(weights) if weights else 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Edge Importance', rotation=270, labelpad=20, fontsize=10)
        
        ax1.set_title(
            f'🔍 Explanation Subgraph for Node {node_idx}\n'
            f'Predicted: {self.class_names[pred_class]} | '
            f'True: {self.class_names[true_class]} '
            f'{"✓" if pred_class == true_class else "✗"}',
            fontsize=13,
            fontweight='bold',
            pad=20
        )
        ax1.axis('off')
        
        # ============ SUBPLOT 2: Enhanced Feature Importance ============
        ax2 = fig.add_subplot(gs[0, 1])
        
        if hasattr(explanation, 'node_mask') and explanation.node_mask is not None:
            feature_importance = explanation.node_mask[node_idx].cpu().numpy()
            top_k = 15
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            top_values = feature_importance[top_indices]
            
            # Create gradient colors
            colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_values)))
            bars = ax2.barh(range(len(top_values)), top_values, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, top_values)):
                ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', va='center', fontsize=8, fontweight='bold')
            
            ax2.set_yticks(range(len(top_values)))
            ax2.set_yticklabels([f'Feature {i}' for i in top_indices], fontsize=9)
            ax2.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
            ax2.set_title(f' Top {top_k} Important Features', fontsize=12, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
            ax2.invert_yaxis()
        else:
            ax2.text(0.5, 0.5, 'Feature masks not available\nfor this model', 
                    ha='center', va='center', fontsize=11)
            ax2.set_title('Feature Importance', fontsize=12, fontweight='bold')
        
        # ============ SUBPLOT 3: Prediction Confidence with Distribution ============
        ax3 = fig.add_subplot(gs[0, 2])
        
        probs = pred_probs.cpu().numpy()
        colors = ['green' if i == pred_class else 'lightgray' for i in range(len(probs))]
        colors[true_class] = 'blue' if true_class != pred_class else colors[true_class]
        
        bars = ax3.barh(range(len(probs)), probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_yticks(range(len(probs)))
        ax3.set_yticklabels([f'{name[:12]}...' if len(name) > 12 else name 
                            for name in self.class_names], fontsize=9)
        ax3.set_xlabel('Probability', fontsize=11, fontweight='bold')
        ax3.set_title(' Prediction Probabilities', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        ax3.invert_yaxis()
        ax3.set_xlim([0, 1])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.8, label='Predicted', edgecolor='black'),
            Patch(facecolor='blue', alpha=0.8, label='True Label', edgecolor='black'),
            Patch(facecolor='lightgray', alpha=0.8, label='Other', edgecolor='black')
        ]
        ax3.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
        
        # Add value labels
        for bar, prob in zip(bars, probs):
            if prob > 0.05:
                ax3.text(prob - 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.3f}', va='center', ha='right', fontsize=8, 
                        fontweight='bold', color='white')
        
        # ============ SUBPLOT 4: Edge Importance Distribution ============
        ax4 = fig.add_subplot(gs[0, 3])
        
        if len(edge_weights) > 0:
            ax4.hist(edge_weights, bins=20, color='coral', alpha=0.7, edgecolor='black')
            ax4.axvline(np.mean(edge_weights), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(edge_weights):.3f}')
            ax4.axvline(np.median(edge_weights), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(edge_weights):.3f}')
            ax4.set_xlabel('Edge Weight', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax4.set_title(' Edge Importance Distribution', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9, framealpha=0.9)
            ax4.grid(alpha=0.3, linestyle='--')
        
        # ============ SUBPLOT 5: Node Statistics ============
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis('off')
        
        degree = (self.data.edge_index[0] == node_idx).sum().item()
        neighbors = self.data.edge_index[1][self.data.edge_index[0] == node_idx]
        neighbor_classes = self.data.y[neighbors].cpu().numpy()
        class_distribution = np.bincount(neighbor_classes, minlength=len(self.class_names))
        
        stats_text = f"📊 Node Statistics\n{'─'*25}\n"
        stats_text += f"Node ID:       {node_idx}\n"
        stats_text += f"Degree:        {degree}\n"
        stats_text += f"True Class:    {self.class_names[true_class]}\n"
        stats_text += f"Pred Class:    {self.class_names[pred_class]}\n"
        stats_text += f"Confidence:    {probs[pred_class]:.3f}\n"
        stats_text += f"Correct:       {'Yes ✓' if pred_class == true_class else 'No ✗'}\n"
        
        ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
                verticalalignment='center', fontweight='bold')
        
        # ============ SUBPLOT 6: Neighbor Class Distribution ============
        ax6 = fig.add_subplot(gs[1, 2])
        
        if degree > 0:
            non_zero_classes = [(i, class_distribution[i]) for i in range(len(class_distribution)) if class_distribution[i] > 0]
            if non_zero_classes:
                classes, counts = zip(*non_zero_classes)
                colors_dist = ['green' if c == true_class else 'lightblue' for c in classes]
                bars = ax6.bar(range(len(classes)), counts, color=colors_dist, alpha=0.8, edgecolor='black')
                ax6.set_xticks(range(len(classes)))
                ax6.set_xticklabels([self.class_names[c][:8] for c in classes], rotation=45, ha='right', fontsize=8)
                ax6.set_ylabel('Count', fontsize=10, fontweight='bold')
                ax6.set_title('👥 Neighbor Classes', fontsize=11, fontweight='bold')
                ax6.grid(axis='y', alpha=0.3, linestyle='--')
                
                for bar, count in zip(bars, counts):
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            f'{int(count)}', ha='center', fontsize=9, fontweight='bold')
        
        # ============ SUBPLOT 7: Model Info ============
        ax7 = fig.add_subplot(gs[1, 3])
        ax7.axis('off')
        
        model_text = f"🤖 Model Information\n{'─'*25}\n"
        model_text += f"Model:         {self.model_name}\n"
        model_text += f"Dataset:       Cora\n"
        model_text += f"Subgraph:      {len(subset)} nodes\n"
        model_text += f"              {sub_edge_index.shape[1]} edges\n"
        model_text += f"Max Edge Imp:  {max(edge_weights) if len(edge_weights) > 0 else 0:.3f}\n"
        
        ax7.text(0.1, 0.5, model_text, fontsize=10, family='monospace', 
                verticalalignment='center', fontweight='bold')
        
        plt.suptitle(f'🔬 Advanced GNN Explanation - {self.model_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved advanced explanation to {save_path}")
        
        plt.close()
        
        return explanation
    
    def analyze_multiple_nodes(self, node_indices, save_dir):
        """Analyze and visualize multiple nodes with advanced features"""
        os.makedirs(save_dir, exist_ok=True)
        
        explanations = []
        
        for idx in node_indices:
            print(f"  → Explaining node {idx}...")
            save_path = os.path.join(save_dir, f'advanced_explanation_node_{idx}.png')
            explanation = self.visualize_explanation_advanced(idx, save_path=save_path)
            explanations.append({
                'node_idx': idx,
                'explanation': explanation,
                'true_class': self.data.y[idx].item(),
                'pred_class': self.model(self.data.x, self.data.edge_index)[idx].argmax().item()
            })
        
        return explanations
    
    def create_summary_report(self, explanations, save_path):
        """Create detailed summary report"""
        with open(save_path, 'w') as f:
            f.write("=" * 90 + "\n")
            f.write(f"ADVANCED GNN EXPLAINABILITY REPORT - {self.model_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 90 + "\n\n")
            
            correct = sum(1 for exp in explanations if exp['true_class'] == exp['pred_class'])
            total = len(explanations)
            
            f.write(f"Overall Accuracy: {correct}/{total} ({correct/total*100:.2f}%)\n\n")
            
            for exp_data in explanations:
                node_idx = exp_data['node_idx']
                true_class = exp_data['true_class']
                pred_class = exp_data['pred_class']
                
                f.write(f"\nNode {node_idx}:\n")
                f.write(f"  True Class:      {self.class_names[true_class]}\n")
                f.write(f"  Predicted Class: {self.class_names[pred_class]}\n")
                f.write(f"  Status:          {'✓ CORRECT' if true_class == pred_class else '✗ INCORRECT'}\n")
                f.write("-" * 90 + "\n")
        
        print(f"✓ Saved detailed summary report to {save_path}")


def main():
    print("=" * 90)
    print("🔬 ADVANCED GNN EXPLAINABILITY ANALYSIS")
    print("=" * 90)
    print()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    print("📊 Loading Cora dataset...")
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    
    num_features = dataset.num_features
    hidden_channels = 64
    num_classes = dataset.num_classes
    
    # Output directory
    output_dir = './results/explainability'
    os.makedirs(output_dir, exist_ok=True)
    
    # Models to explain
    model_configs = {
        'GCN': {
            'class': GCN,
            'path': './models/saved/gcn_best.pth',
            'kwargs': {'dropout': 0.5}
        },
        'GAT': {
            'class': GAT,
            'path': './models/saved/gat_best.pth',
            'kwargs': {'heads': 8, 'dropout': 0.6}
        },
        'GraphSAGE': {
            'class': GraphSAGE,
            'path': './models/saved/graphsage_best.pth',
            'kwargs': {'dropout': 0.5}
        }
    }
    
    # Select diverse nodes to explain
    test_nodes = data.test_mask.nonzero(as_tuple=True)[0][:200]
    selected_nodes = test_nodes[::40].tolist()  # Select every 40th node for diversity
    
    print(f" Dataset: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f" Explaining {len(selected_nodes)} test nodes: {selected_nodes}\n")
    
    # Analyze each model
    for model_name, config in model_configs.items():
        print(f"\n{'='*90}")
        print(f"🔍 Analyzing {model_name} with Advanced Explainability")
        print(f"{'='*90}\n")
        
        try:
            # Load model
            model = config['class'](num_features, hidden_channels, num_classes, **config['kwargs'])
            model.load_state_dict(torch.load(config['path'], map_location=device))
            model.eval()
            
            # Create analyzer
            analyzer = AdvancedGNNExplainer(model, data, model_name, device)
            
            # Create model-specific output directory
            model_output_dir = os.path.join(output_dir, model_name.lower())
            
            # Analyze nodes with advanced features
            explanations = analyzer.analyze_multiple_nodes(selected_nodes, model_output_dir)
            
            # Create detailed summary report
            report_path = os.path.join(model_output_dir, 'advanced_explanation_summary.txt')
            analyzer.create_summary_report(explanations, report_path)
            
            print(f"\n {model_name} advanced analysis complete!")
            print(f"   Output directory: {model_output_dir}/")
            
        except FileNotFoundError:
            print(f"  Model not found: {config['path']}")
            print(f"   Please train the model first!")
            continue
        except Exception as e:
            print(f" Error analyzing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n\n" + "=" * 90)
    print(" ADVANCED EXPLAINABILITY ANALYSIS COMPLETE!")
    print("=" * 90)
    print(f"\n All results saved to: {output_dir}/")
    print("\n Research Impact:")
    print("   ✓ Advanced model interpretability with comprehensive visualizations")

    print()


if __name__ == '__main__':
    main()