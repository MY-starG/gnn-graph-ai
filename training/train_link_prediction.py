"""
Link Prediction Training for GNN Models

This script trains GNN models for link prediction tasks:
1. Predicts missing edges in the graph
2. Uses negative sampling for training
3. Evaluates with AUC-ROC and Average Precision
4. Demonstrates versatility beyond node classification

Designed for research publication and PhD applications
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import negative_sampling, train_test_split_edges


# ===================== MODEL DEFINITIONS =====================

class LinkPredictorGCN(torch.nn.Module):
    """GCN model for link prediction"""
    def __init__(self, num_features, hidden_channels, dropout=0.5):
        super(LinkPredictorGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        
    def encode(self, x, edge_index):
        """Encode node features"""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        """Decode edge probabilities"""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)


class LinkPredictorGAT(torch.nn.Module):
    """GAT model for link prediction"""
    def __init__(self, num_features, hidden_channels, heads=8, dropout=0.6):
        super(LinkPredictorGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        
    def encode(self, x, edge_index):
        """Encode node features"""
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        """Decode edge probabilities"""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)


class LinkPredictorGraphSAGE(torch.nn.Module):
    """GraphSAGE model for link prediction"""
    def __init__(self, num_features, hidden_channels, dropout=0.5):
        super(LinkPredictorGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        
    def encode(self, x, edge_index):
        """Encode node features"""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        """Decode edge probabilities"""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_index)


# ===================== TRAINING AND EVALUATION =====================

def get_link_labels(pos_edge_index, neg_edge_index):
    """Create labels for positive and negative edges"""
    num_pos = pos_edge_index.size(1)
    num_neg = neg_edge_index.size(1)
    
    link_labels = torch.zeros(num_pos + num_neg, dtype=torch.float)
    link_labels[:num_pos] = 1.0
    
    return link_labels


def train_epoch(model, data, optimizer):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    # Positive edges
    pos_edge_index = data.train_pos_edge_index
    
    # Negative sampling
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )
    
    # Forward pass
    z = model.encode(data.x, pos_edge_index)
    
    # Decode positive and negative edges
    pos_pred = model.decode(z, pos_edge_index)
    neg_pred = model.decode(z, neg_edge_index)
    
    # Concatenate predictions and labels
    pred = torch.cat([pos_pred, neg_pred])
    labels = torch.cat([
        torch.ones(pos_pred.size(0)),
        torch.zeros(neg_pred.size(0))
    ])
    
    # Binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(pred, labels)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model, data, edge_index, neg_edge_index):
    """Evaluate model on validation or test set"""
    model.eval()
    
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)
        
        pos_pred = model.decode(z, edge_index)
        neg_pred = model.decode(z, neg_edge_index)
        
        pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
        labels = torch.cat([
            torch.ones(pos_pred.size(0)),
            torch.zeros(neg_pred.size(0))
        ]).cpu().numpy()
        
        # Compute metrics
        auc = roc_auc_score(labels, pred)
        ap = average_precision_score(labels, pred)
    
    return auc, ap


def train_link_prediction(model, data, epochs=200, lr=0.01, early_stopping=50):
    """Train link prediction model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    best_val_auc = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_auc': [],
        'val_ap': [],
        'test_auc': [],
        'test_ap': []
    }
    
    for epoch in range(epochs):
        # Training
        loss = train_epoch(model, data, optimizer)
        
        # Validation
        val_auc, val_ap = evaluate(model, data, data.val_pos_edge_index, data.val_neg_edge_index)
        test_auc, test_ap = evaluate(model, data, data.test_pos_edge_index, data.test_neg_edge_index)
        
        history['train_loss'].append(loss)
        history['val_auc'].append(val_auc)
        history['val_ap'].append(val_ap)
        history['test_auc'].append(test_auc)
        history['test_ap'].append(test_ap)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc = test_auc
            best_test_ap = test_ap
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')
        
        if patience_counter >= early_stopping:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return {
        'best_val_auc': best_val_auc,
        'best_test_auc': best_test_auc,
        'best_test_ap': best_test_ap,
        'history': history
    }


# ===================== VISUALIZATION =====================

def plot_training_curves(histories, save_path):
    """Plot training curves for all models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # AUC curves
    ax = axes[0]
    for model_name, history in histories.items():
        epochs = range(1, len(history['val_auc']) + 1)
        ax.plot(epochs, history['val_auc'], linewidth=2, label=f'{model_name} (Val)', marker='o', markersize=3)
        ax.plot(epochs, history['test_auc'], linewidth=2, label=f'{model_name} (Test)', marker='s', markersize=3, linestyle='--')
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('AUC-ROC', fontweight='bold')
    ax.set_title('Link Prediction: AUC-ROC Over Training', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    # AP curves
    ax = axes[1]
    for model_name, history in histories.items():
        epochs = range(1, len(history['val_ap']) + 1)
        ax.plot(epochs, history['val_ap'], linewidth=2, label=f'{model_name} (Val)', marker='o', markersize=3)
        ax.plot(epochs, history['test_ap'], linewidth=2, label=f'{model_name} (Test)', marker='s', markersize=3, linestyle='--')
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Average Precision', fontweight='bold')
    ax.set_title('Link Prediction: Average Precision Over Training', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Training curves saved to {save_path}')


def plot_final_comparison(results, save_path):
    """Plot final comparison of all models"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    model_names = list(results.keys())
    test_aucs = [results[m]['best_test_auc'] for m in model_names]
    test_aps = [results[m]['best_test_ap'] for m in model_names]
    
    # AUC comparison
    ax = axes[0]
    colors = ['steelblue', 'coral', 'mediumseagreen']
    bars = ax.bar(model_names, test_aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Test AUC-ROC', fontweight='bold')
    ax.set_title('Link Prediction: Test AUC-ROC', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, auc in zip(bars, test_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{auc:.4f}', ha='center', fontweight='bold')
    
    # AP comparison
    ax = axes[1]
    bars = ax.bar(model_names, test_aps, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Test Average Precision', fontweight='bold')
    ax.set_title('Link Prediction: Test Average Precision', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    for bar, ap in zip(bars, test_aps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{ap:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Comparison plot saved to {save_path}')


def plot_roc_curves(models, data, model_names, save_path):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    colors = ['steelblue', 'coral', 'mediumseagreen']
    
    for model, model_name, color in zip(models, model_names, colors):
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.train_pos_edge_index)
            
            pos_pred = model.decode(z, data.test_pos_edge_index)
            neg_pred = model.decode(z, data.test_neg_edge_index)
            
            pred = torch.cat([pos_pred, neg_pred]).cpu().numpy()
            labels = torch.cat([
                torch.ones(pos_pred.size(0)),
                torch.zeros(neg_pred.size(0))
            ]).cpu().numpy()
            
            fpr, tpr, _ = roc_curve(labels, pred)
            auc = roc_auc_score(labels, pred)
            
            plt.plot(fpr, tpr, linewidth=2.5, label=f'{model_name} (AUC = {auc:.4f})', color=color)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    plt.title('ROC Curves - Link Prediction', fontweight='bold', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'ROC curves saved to {save_path}')


# ===================== MAIN EXECUTION =====================

def main():
    print("="*90)
    print("LINK PREDICTION TRAINING FOR GNN MODELS")
    print("="*90)
    print()
    
    # Load data
    print("Loading Cora dataset...")
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    
    # Split edges for link prediction
    print("Splitting edges for link prediction...")
    data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1)
    
    print(f"Dataset: {data.num_nodes} nodes")
    print(f"Training edges: {data.train_pos_edge_index.size(1)}")
    print(f"Validation edges: {data.val_pos_edge_index.size(1)}")
    print(f"Test edges: {data.test_pos_edge_index.size(1)}")
    print()
    
    num_features = dataset.num_features
    hidden_channels = 64
    
    # Output directory
    output_dir = './results/link_prediction'
    os.makedirs(output_dir, exist_ok=True)
    
    # Model configurations
    model_configs = {
        'GCN': {
            'class': LinkPredictorGCN,
            'kwargs': {'num_features': num_features, 'hidden_channels': hidden_channels, 'dropout': 0.5}
        },
        'GAT': {
            'class': LinkPredictorGAT,
            'kwargs': {'num_features': num_features, 'hidden_channels': hidden_channels, 'heads': 8, 'dropout': 0.6}
        },
        'GraphSAGE': {
            'class': LinkPredictorGraphSAGE,
            'kwargs': {'num_features': num_features, 'hidden_channels': hidden_channels, 'dropout': 0.5}
        }
    }
    
    # Train all models
    results = {}
    histories = {}
    trained_models = []
    
    for model_name, config in model_configs.items():
        print("\n" + "="*90)
        print(f"Training {model_name}")
        print("="*90)
        
        model = config['class'](**config['kwargs'])
        result = train_link_prediction(model, data, epochs=200, lr=0.01, early_stopping=50)
        
        results[model_name] = result
        histories[model_name] = result['history']
        trained_models.append(model)
        
        print(f"\n{model_name} Results:")
        print(f"  Best Val AUC: {result['best_val_auc']:.4f}")
        print(f"  Best Test AUC: {result['best_test_auc']:.4f}")
        print(f"  Best Test AP: {result['best_test_ap']:.4f}")
    
    # Generate visualizations
    print("\n" + "="*90)
    print("GENERATING VISUALIZATIONS")
    print("="*90)
    
    curves_path = os.path.join(output_dir, 'training_curves.png')
    plot_training_curves(histories, curves_path)
    
    comparison_path = os.path.join(output_dir, 'final_comparison.png')
    plot_final_comparison(results, comparison_path)
    
    roc_path = os.path.join(output_dir, 'roc_curves.png')
    plot_roc_curves(trained_models, data, list(model_configs.keys()), roc_path)
    
    # Create results table
    print("\nCreating results table...")
    import pandas as pd
    
    table_data = []
    for model_name, result in results.items():
        table_data.append({
            'Model': model_name,
            'Val AUC': f"{result['best_val_auc']:.4f}",
            'Test AUC': f"{result['best_test_auc']:.4f}",
            'Test AP': f"{result['best_test_ap']:.4f}"
        })
    
    df = pd.DataFrame(table_data)
    
    csv_path = os.path.join(output_dir, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f'Results table saved to {csv_path}')
    
    # Create results visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    best_auc_idx = df['Test AUC'].astype(float).idxmax()
    best_ap_idx = df['Test AP'].astype(float).idxmax()
    table[(best_auc_idx + 1, 2)].set_facecolor('#FFD700')
    table[(best_ap_idx + 1, 3)].set_facecolor('#FFD700')
    
    plt.title('Link Prediction Results', fontsize=14, fontweight='bold', pad=20)
    
    table_img_path = os.path.join(output_dir, 'results_table.png')
    plt.savefig(table_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Results table visualization saved to {table_img_path}')
    
    print("\n" + "="*90)
    print("LINK PREDICTION TRAINING COMPLETE")
    print("="*90)
    print(f"\nAll results saved to: {output_dir}")
    print("\nGenerated outputs:")
    print("  - Training curves (PNG)")
    print("  - Final comparison (PNG)")
    print("  - ROC curves (PNG)")
    print("  - Results table (CSV + PNG)")
    print("\nDemonstrates versatility beyond node classification!")
    print()


if __name__ == '__main__':
    main()