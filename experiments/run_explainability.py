"""
Advanced Explainability Demo Script for GNN Models

This script provides comprehensive model interpretability analysis by:
1. Generating explanations for diverse test nodes
2. Analyzing feature importance patterns
3. Identifying critical subgraph structures
4. Comparing explainability across models
5. Creating publication-quality visualizations
6. Statistical analysis of explanation patterns

Designed for research publication and Erasmus Mundus PhD applications
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
import json
import pandas as pd
from collections import defaultdict

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph, degree
import networkx as nx


# ===================== MODEL DEFINITIONS =====================

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


# ===================== EXPLAINABILITY ANALYZER =====================

class AdvancedExplainabilityAnalyzer:
    """Comprehensive explainability analysis for GNN models"""
    
    def __init__(self, model, data, model_name, device='cpu'):
        self.model = model.to(device)
        self.data = data.to(device)
        self.model_name = model_name
        self.device = device
        self.model.eval()
        
        # Initialize explainer
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
        
        self.class_names = [
            'Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
            'Probabilistic_Methods', 'Reinforcement_Learning',
            'Rule_Learning', 'Theory'
        ]
        
        # Storage for analysis
        self.explanations_data = []
        
        print(f"Advanced Explainability Analyzer initialized for {model_name}")
    
    def explain_node(self, node_idx):
        """Generate explanation for a single node"""
        explanation = self.explainer(
            self.data.x,
            self.data.edge_index,
            index=node_idx
        )
        return explanation
    
    def analyze_node_comprehensive(self, node_idx):
        """Comprehensive analysis of a single node"""
        explanation = self.explain_node(node_idx)
        
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred_class = out[node_idx].argmax().item()
            true_class = self.data.y[node_idx].item()
            pred_probs = torch.softmax(out[node_idx], dim=0)
            confidence = pred_probs[pred_class].item()
        
        # Extract explanation components
        edge_mask = explanation.edge_mask.cpu().numpy()
        node_mask = explanation.node_mask[node_idx].cpu().numpy() if hasattr(explanation, 'node_mask') and explanation.node_mask is not None else None
        
        # Compute statistics
        analysis = {
            'node_idx': node_idx,
            'true_class': true_class,
            'pred_class': pred_class,
            'confidence': confidence,
            'correct': true_class == pred_class,
            'edge_weights_mean': float(np.mean(edge_mask)),
            'edge_weights_std': float(np.std(edge_mask)),
            'edge_weights_max': float(np.max(edge_mask)),
            'num_important_edges': int(np.sum(edge_mask > np.percentile(edge_mask, 75))),
            'node_degree': int(degree(self.data.edge_index[0], num_nodes=self.data.num_nodes)[node_idx].item())
        }
        
        if node_mask is not None:
            analysis['feature_importance_mean'] = float(np.mean(node_mask))
            analysis['feature_importance_max'] = float(np.max(node_mask))
            analysis['num_important_features'] = int(np.sum(node_mask > np.percentile(node_mask, 75)))
        
        self.explanations_data.append(analysis)
        return analysis, explanation
    
    def select_diverse_nodes(self, num_nodes=20):
        """Select diverse nodes for explanation based on different criteria"""
        test_mask = self.data.test_mask
        test_indices = torch.where(test_mask)[0]
        
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            predictions = out.argmax(dim=1)
            probabilities = torch.softmax(out, dim=1)
        
        selected_nodes = []
        
        # 1. High confidence correct predictions
        correct_mask = predictions[test_mask] == self.data.y[test_mask]
        correct_indices = test_indices[correct_mask]
        if len(correct_indices) > 0:
            confidences = probabilities[correct_indices].max(dim=1)[0]
            top_confident = correct_indices[torch.topk(confidences, min(5, len(confidences)))[1]]
            selected_nodes.extend(top_confident.tolist())
        
        # 2. Low confidence correct predictions
        if len(correct_indices) > 0:
            confidences = probabilities[correct_indices].max(dim=1)[0]
            low_confident = correct_indices[torch.topk(confidences, min(5, len(confidences)), largest=False)[1]]
            selected_nodes.extend(low_confident.tolist())
        
        # 3. Incorrect predictions
        incorrect_mask = ~correct_mask
        incorrect_indices = test_indices[incorrect_mask]
        if len(incorrect_indices) > 0:
            selected_nodes.extend(incorrect_indices[:min(5, len(incorrect_indices))].tolist())
        
        # 4. Different degree nodes
        degrees = degree(self.data.edge_index[0], num_nodes=self.data.num_nodes)[test_indices]
        for percentile in [25, 50, 75]:
            threshold = np.percentile(degrees.cpu().numpy(), percentile)
            candidates = test_indices[torch.abs(degrees - threshold) < 2]
            if len(candidates) > 0:
                selected_nodes.append(candidates[0].item())
        
        # Remove duplicates and limit
        selected_nodes = list(set(selected_nodes))[:num_nodes]
        
        return selected_nodes
    
    def generate_comparative_visualization(self, node_idx, save_path):
        """Generate detailed comparative visualization for a node"""
        analysis, explanation = self.analyze_node_comprehensive(node_idx)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # Get subgraph
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx, 2, self.data.edge_index, relabel_nodes=True, num_nodes=self.data.num_nodes
        )
        
        # 1. Subgraph visualization
        ax1 = fig.add_subplot(gs[:2, 0])
        G = nx.Graph()
        for i, node in enumerate(subset):
            G.add_node(i)
        
        edge_mask = explanation.edge_mask[:len(sub_edge_index[0])].cpu().numpy()
        for i in range(sub_edge_index.shape[1]):
            src, dst = sub_edge_index[:, i].cpu().numpy()
            weight = edge_mask[i] if i < len(edge_mask) else 0.0
            G.add_edge(src, dst, weight=weight)
        
        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
        
        node_colors = [self.data.y[subset[i]].item() for i in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, 
                              cmap='tab10', alpha=0.8, ax=ax1, edgecolors='black', linewidths=1.5)
        
        nx.draw_networkx_nodes(G, pos, nodelist=[mapping.item()], node_color='red',
                              node_size=800, node_shape='*', ax=ax1, edgecolors='darkred', linewidths=2)
        
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        if max(weights) > 0:
            normalized_weights = [w / max(weights) for w in weights]
        else:
            normalized_weights = weights
        
        nx.draw_networkx_edges(G, pos, width=[w * 6 for w in normalized_weights],
                              edge_color=weights, edge_cmap=plt.cm.Reds,
                              alpha=0.6, ax=ax1)
        
        ax1.set_title(f'Explanation Subgraph - Node {node_idx}\n'
                     f'Prediction: {self.class_names[analysis["pred_class"]]} '
                     f'(Confidence: {analysis["confidence"]:.3f})',
                     fontweight='bold', fontsize=11)
        ax1.axis('off')
        
        # 2. Edge weight distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(edge_mask, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(edge_mask), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(edge_mask):.3f}')
        ax2.set_xlabel('Edge Weight', fontweight='bold')
        ax2.set_ylabel('Frequency', fontweight='bold')
        ax2.set_title('Edge Importance Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Top important edges
        ax3 = fig.add_subplot(gs[1, 1])
        top_k = min(10, len(edge_mask))
        top_indices = np.argsort(edge_mask)[-top_k:][::-1]
        top_weights = edge_mask[top_indices]
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_weights)))
        bars = ax3.barh(range(len(top_weights)), top_weights, color=colors, 
                       alpha=0.8, edgecolor='black')
        ax3.set_yticks(range(len(top_weights)))
        ax3.set_yticklabels([f'Edge {i}' for i in top_indices])
        ax3.set_xlabel('Importance Score', fontweight='bold')
        ax3.set_title(f'Top {top_k} Important Edges', fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        ax3.invert_yaxis()
        
        # 4. Prediction probabilities
        ax4 = fig.add_subplot(gs[0, 2])
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            probs = torch.softmax(out[node_idx], dim=0).cpu().numpy()
        
        colors_pred = ['green' if i == analysis['pred_class'] else 'lightgray' 
                      for i in range(len(probs))]
        colors_pred[analysis['true_class']] = 'blue' if analysis['true_class'] != analysis['pred_class'] else colors_pred[analysis['true_class']]
        
        bars = ax4.barh(range(len(probs)), probs, color=colors_pred, 
                       alpha=0.8, edgecolor='black')
        ax4.set_yticks(range(len(probs)))
        ax4.set_yticklabels([name[:12] for name in self.class_names], fontsize=9)
        ax4.set_xlabel('Probability', fontweight='bold')
        ax4.set_title('Class Probabilities', fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        ax4.invert_yaxis()
        
        # 5. Feature importance (if available)
        ax5 = fig.add_subplot(gs[1, 2])
        if hasattr(explanation, 'node_mask') and explanation.node_mask is not None:
            node_mask = explanation.node_mask[node_idx].cpu().numpy()
            top_k_feat = min(15, len(node_mask))
            top_feat_indices = np.argsort(node_mask)[-top_k_feat:][::-1]
            top_feat_values = node_mask[top_feat_indices]
            
            colors_feat = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_feat_values)))
            bars = ax5.barh(range(len(top_feat_values)), top_feat_values, 
                          color=colors_feat, alpha=0.8, edgecolor='black')
            ax5.set_yticks(range(len(top_feat_values)))
            ax5.set_yticklabels([f'F{i}' for i in top_feat_indices], fontsize=8)
            ax5.set_xlabel('Importance', fontweight='bold')
            ax5.set_title(f'Top {top_k_feat} Features', fontweight='bold')
            ax5.grid(axis='x', alpha=0.3)
            ax5.invert_yaxis()
        else:
            ax5.text(0.5, 0.5, 'Feature importance\nnot available', 
                    ha='center', va='center', fontsize=11)
            ax5.set_title('Feature Importance', fontweight='bold')
        
        # 6. Node statistics
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.axis('off')
        stats_text = "Node Statistics\n" + "="*30 + "\n"
        stats_text += f"Node ID: {node_idx}\n"
        stats_text += f"Degree: {analysis['node_degree']}\n"
        stats_text += f"True Class: {self.class_names[analysis['true_class']]}\n"
        stats_text += f"Predicted: {self.class_names[analysis['pred_class']]}\n"
        stats_text += f"Confidence: {analysis['confidence']:.4f}\n"
        stats_text += f"Status: {'CORRECT' if analysis['correct'] else 'INCORRECT'}\n"
        ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
                verticalalignment='center', fontweight='bold')
        
        # 7. Explanation statistics
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.axis('off')
        exp_text = "Explanation Statistics\n" + "="*30 + "\n"
        exp_text += f"Edge Weights:\n"
        exp_text += f"  Mean: {analysis['edge_weights_mean']:.4f}\n"
        exp_text += f"  Std: {analysis['edge_weights_std']:.4f}\n"
        exp_text += f"  Max: {analysis['edge_weights_max']:.4f}\n"
        exp_text += f"Important Edges: {analysis['num_important_edges']}\n"
        if 'feature_importance_mean' in analysis:
            exp_text += f"Feature Importance:\n"
            exp_text += f"  Mean: {analysis['feature_importance_mean']:.4f}\n"
            exp_text += f"  Max: {analysis['feature_importance_max']:.4f}\n"
        ax7.text(0.1, 0.5, exp_text, fontsize=10, family='monospace',
                verticalalignment='center', fontweight='bold')
        
        # 8. Neighborhood analysis
        ax8 = fig.add_subplot(gs[2, 2])
        neighbors = self.data.edge_index[1][self.data.edge_index[0] == node_idx]
        if len(neighbors) > 0:
            neighbor_classes = self.data.y[neighbors].cpu().numpy()
            class_counts = np.bincount(neighbor_classes, minlength=len(self.class_names))
            non_zero_classes = np.where(class_counts > 0)[0]
            
            if len(non_zero_classes) > 0:
                colors_neigh = ['green' if c == analysis['true_class'] else 'lightblue' 
                               for c in non_zero_classes]
                bars = ax8.bar(range(len(non_zero_classes)), 
                              class_counts[non_zero_classes],
                              color=colors_neigh, alpha=0.8, edgecolor='black')
                ax8.set_xticks(range(len(non_zero_classes)))
                ax8.set_xticklabels([self.class_names[c][:8] for c in non_zero_classes],
                                   rotation=45, ha='right', fontsize=8)
                ax8.set_ylabel('Count', fontweight='bold')
                ax8.set_title('Neighbor Classes', fontweight='bold')
                ax8.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Comprehensive Explanation Analysis - {self.model_name} - Node {node_idx}',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return analysis
    
    def generate_statistical_report(self, save_path):
        """Generate statistical analysis report of all explanations"""
        if not self.explanations_data:
            print("No explanation data available")
            return
        
        df = pd.DataFrame(self.explanations_data)
        
        with open(save_path, 'w') as f:
            f.write("="*90 + "\n")
            f.write(f"STATISTICAL EXPLAINABILITY REPORT - {self.model_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*90 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*90 + "\n")
            f.write(f"Total nodes analyzed: {len(df)}\n")
            f.write(f"Correct predictions: {df['correct'].sum()} ({df['correct'].mean()*100:.2f}%)\n")
            f.write(f"Mean confidence: {df['confidence'].mean():.4f} +/- {df['confidence'].std():.4f}\n\n")
            
            # Edge importance statistics
            f.write("EDGE IMPORTANCE STATISTICS\n")
            f.write("-"*90 + "\n")
            f.write(f"Mean edge weight: {df['edge_weights_mean'].mean():.4f} +/- {df['edge_weights_mean'].std():.4f}\n")
            f.write(f"Max edge weight: {df['edge_weights_max'].mean():.4f} +/- {df['edge_weights_max'].std():.4f}\n")
            f.write(f"Mean important edges: {df['num_important_edges'].mean():.2f} +/- {df['num_important_edges'].std():.2f}\n\n")
            
            # Correlation analysis
            f.write("CORRELATION ANALYSIS\n")
            f.write("-"*90 + "\n")
            corr_conf_edge = df['confidence'].corr(df['edge_weights_mean'])
            f.write(f"Confidence vs Edge Weights: {corr_conf_edge:.4f}\n")
            corr_degree_edge = df['node_degree'].corr(df['num_important_edges'])
            f.write(f"Node Degree vs Important Edges: {corr_degree_edge:.4f}\n\n")
            
            # Correct vs Incorrect predictions
            f.write("CORRECT VS INCORRECT PREDICTIONS\n")
            f.write("-"*90 + "\n")
            correct_df = df[df['correct'] == True]
            incorrect_df = df[df['correct'] == False]
            
            if len(correct_df) > 0:
                f.write(f"Correct predictions:\n")
                f.write(f"  Mean confidence: {correct_df['confidence'].mean():.4f}\n")
                f.write(f"  Mean edge weight: {correct_df['edge_weights_mean'].mean():.4f}\n")
                f.write(f"  Mean important edges: {correct_df['num_important_edges'].mean():.2f}\n\n")
            
            if len(incorrect_df) > 0:
                f.write(f"Incorrect predictions:\n")
                f.write(f"  Mean confidence: {incorrect_df['confidence'].mean():.4f}\n")
                f.write(f"  Mean edge weight: {incorrect_df['edge_weights_mean'].mean():.4f}\n")
                f.write(f"  Mean important edges: {incorrect_df['num_important_edges'].mean():.2f}\n\n")
        
        print(f"Statistical report saved to {save_path}")
    
    def create_summary_visualizations(self, save_dir):
        """Create summary visualizations across all analyzed nodes"""
        if not self.explanations_data:
            print("No explanation data available")
            return
        
        df = pd.DataFrame(self.explanations_data)
        
        # Figure 1: Statistical distributions
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Confidence distribution
        axes[0, 0].hist(df['confidence'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(df['confidence'].mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {df["confidence"].mean():.3f}')
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Prediction Confidence Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Edge weights distribution
        axes[0, 1].hist(df['edge_weights_mean'], bins=20, color='coral', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(df['edge_weights_mean'].mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {df["edge_weights_mean"].mean():.3f}')
        axes[0, 1].set_xlabel('Mean Edge Weight')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Edge Importance Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Important edges vs degree
        axes[0, 2].scatter(df['node_degree'], df['num_important_edges'], 
                          c=df['confidence'], cmap='viridis', s=100, alpha=0.6, edgecolors='black')
        axes[0, 2].set_xlabel('Node Degree')
        axes[0, 2].set_ylabel('Number of Important Edges')
        axes[0, 2].set_title('Degree vs Important Edges')
        axes[0, 2].grid(alpha=0.3)
        cbar = plt.colorbar(axes[0, 2].collections[0], ax=axes[0, 2])
        cbar.set_label('Confidence')
        
        # Confidence vs edge weights
        correct_mask = df['correct'] == True
        axes[1, 0].scatter(df[correct_mask]['confidence'], df[correct_mask]['edge_weights_mean'],
                          label='Correct', alpha=0.6, s=100, edgecolors='black')
        axes[1, 0].scatter(df[~correct_mask]['confidence'], df[~correct_mask]['edge_weights_mean'],
                          label='Incorrect', alpha=0.6, s=100, edgecolors='black')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Mean Edge Weight')
        axes[1, 0].set_title('Confidence vs Edge Importance')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Correct vs Incorrect comparison
        correct_stats = [df[correct_mask]['confidence'].mean(),
                        df[correct_mask]['edge_weights_mean'].mean(),
                        df[correct_mask]['num_important_edges'].mean()]
        incorrect_stats = [df[~correct_mask]['confidence'].mean(),
                          df[~correct_mask]['edge_weights_mean'].mean(),
                          df[~correct_mask]['num_important_edges'].mean()] if (~correct_mask).sum() > 0 else [0, 0, 0]
        
        x_pos = np.arange(3)
        width = 0.35
        axes[1, 1].bar(x_pos - width/2, correct_stats, width, label='Correct', 
                      alpha=0.8, color='green', edgecolor='black')
        axes[1, 1].bar(x_pos + width/2, incorrect_stats, width, label='Incorrect',
                      alpha=0.8, color='red', edgecolor='black')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(['Confidence', 'Edge Weight', 'Important Edges'], rotation=15)
        axes[1, 1].set_ylabel('Mean Value')
        axes[1, 1].set_title('Correct vs Incorrect Predictions')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Accuracy by confidence bins
        bins = [0, 0.5, 0.7, 0.9, 1.0]
        bin_labels = ['0-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0']
        bin_accuracies = []
        for i in range(len(bins)-1):
            mask = (df['confidence'] >= bins[i]) & (df['confidence'] < bins[i+1])
            if mask.sum() > 0:
                bin_accuracies.append(df[mask]['correct'].mean())
            else:
                bin_accuracies.append(0)
        
        axes[1, 2].bar(range(len(bin_accuracies)), bin_accuracies, 
                      alpha=0.8, color='purple', edgecolor='black')
        axes[1, 2].set_xticks(range(len(bin_accuracies)))
        axes[1, 2].set_xticklabels(bin_labels)
        axes[1, 2].set_xlabel('Confidence Range')
        axes[1, 2].set_ylabel('Accuracy')
        axes[1, 2].set_title('Accuracy by Confidence Level')
        axes[1, 2].set_ylim([0, 1])
        axes[1, 2].grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Explainability Statistical Analysis - {self.model_name}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'{self.model_name.lower()}_explainability_statistics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary visualizations saved to {save_path}")


# ===================== MAIN EXECUTION =====================

def main():
    print("="*90)
    print("ADVANCED EXPLAINABILITY DEMO - PUBLICATION QUALITY")
    print("="*90)
    print()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading Cora dataset...")
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    
    num_features = dataset.num_features
    hidden_channels = 64
    num_classes = dataset.num_classes
    
    # Output directories
    base_output_dir = './results/explainability_demo'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Model configurations
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
    
    # Process each model
    for model_name, config in model_configs.items():
        print(f"\n{'='*90}")
        print(f"Processing {model_name}")
        print(f"{'='*90}\n")
        
        try:
            # Load model
            model = config['class'](num_features, hidden_channels, num_classes, **config['kwargs'])
            model.load_state_dict(torch.load(config['path'], map_location=device))
            model.eval()
            print(f"Model loaded successfully from {config['path']}")
            
            # Create analyzer
            analyzer = AdvancedExplainabilityAnalyzer(model, data, model_name, device)
            
            # Select diverse nodes
            print("\nSelecting diverse nodes for analysis...")
            selected_nodes = analyzer.select_diverse_nodes(num_nodes=20)
            print(f"Selected {len(selected_nodes)} nodes: {selected_nodes[:10]}...")
            
            # Create model-specific directory
            model_output_dir = os.path.join(base_output_dir, model_name.lower())
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Generate explanations
            print(f"\nGenerating explanations for {len(selected_nodes)} nodes...")
            for idx, node_idx in enumerate(selected_nodes):
                print(f"  [{idx+1}/{len(selected_nodes)}] Explaining node {node_idx}...")
                save_path = os.path.join(model_output_dir, f'explanation_node_{node_idx}.png')
                analyzer.generate_comparative_visualization(node_idx, save_path)
            
            # Generate statistical report
            print("\nGenerating statistical report...")
            report_path = os.path.join(model_output_dir, 'statistical_report.txt')
            analyzer.generate_statistical_report(report_path)
            
            # Generate summary visualizations
            print("Generating summary visualizations...")
            analyzer.create_summary_visualizations(model_output_dir)
            
            # Save explanation data
            data_path = os.path.join(model_output_dir, 'explanations_data.json')
            with open(data_path, 'w') as f:
                json.dump(analyzer.explanations_data, f, indent=2)
            print(f"Explanation data saved to {data_path}")
            
            print(f"\n{model_name} processing complete!")
            print(f"Output directory: {model_output_dir}")
            
        except FileNotFoundError:
            print(f"ERROR: Model not found at {config['path']}")
            print("Please train the model first!")
            continue
        except Exception as e:
            print(f"ERROR processing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*90)
    print("EXPLAINABILITY DEMO COMPLETE")
    print("="*90)
    print(f"\nAll results saved to: {base_output_dir}/")
    print("\nGenerated outputs:")
    print("  - Individual node explanations (PNG)")
    print("  - Statistical reports (TXT)")
    print("  - Summary visualizations (PNG)")
    print("  - Explanation data (JSON)")
    print("\nPublication-ready materials for Erasmus Mundus application!")
    print()


if __name__ == '__main__':
    main()