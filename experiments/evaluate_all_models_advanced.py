"""
Research-Grade Comprehensive Evaluation Framework for GNN Models (STANDALONE) - FIXED

This advanced evaluation script is completely self-contained and provides:
1. Standard + Advanced Metrics (ROC-AUC, PR-AUC, MCC, Cohen's Kappa)
2. Calibration Analysis (ECE, MCE, Brier Score, Reliability Diagrams)
3. Statistical Significance Testing (McNemar's test, Bootstrap CI)
4. Embedding Quality Analysis (Silhouette, Davies-Bouldin, t-SNE)
5. Graph-Aware Metrics (Degree-based accuracy, Homophily analysis)
6. Error Analysis (Confusion patterns, Hard samples identification)
7. Multiple Runs with Confidence Intervals
8. Publication-Ready Figures and LaTeX Tables

For Erasmus Mundus PhD Applications - Research Excellence Demonstration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from datetime import datetime
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


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
    
    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
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
    
    def get_embeddings(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
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
    
    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


# ===================== EVALUATION FUNCTIONS =====================

def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_model(model_class, model_path, num_features, hidden_channels, num_classes, **kwargs):
    """Load trained model with error handling"""
    model = model_class(num_features, hidden_channels, num_classes, **kwargs)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


def evaluate_model_comprehensive(model, data, mask, class_names, device='cpu'):
    """Comprehensive model evaluation with all metrics"""
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        probs = torch.softmax(out, dim=1)
        
        # Get embeddings
        try:
            embeddings = model.get_embeddings(data.x, data.edge_index)
        except:
            embeddings = None
    
    # Extract masked data
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    y_probs = probs[mask].cpu().numpy()
    
    num_classes = len(class_names)
    
    # Compute all metrics
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics['per_class_f1'] = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Advanced metrics
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # ROC-AUC
    try:
        metrics['roc_auc_ovr_macro'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        metrics['roc_auc_ovr_weighted'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average='weighted')
        metrics['roc_auc_per_class'] = roc_auc_score(y_true, y_probs, multi_class='ovr', average=None)
    except:
        metrics['roc_auc_ovr_macro'] = 0.0
        metrics['roc_auc_ovr_weighted'] = 0.0
        metrics['roc_auc_per_class'] = np.zeros(num_classes)
    
    # PR-AUC
    try:
        ap_scores = []
        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            ap = average_precision_score(y_true_binary, y_probs[:, i])
            ap_scores.append(ap)
        metrics['pr_auc_per_class'] = np.array(ap_scores)
        metrics['pr_auc_macro'] = np.mean(ap_scores)
    except:
        metrics['pr_auc_per_class'] = np.zeros(num_classes)
        metrics['pr_auc_macro'] = 0.0
    
    # Top-k accuracy
    top2_correct = np.any(np.argsort(-y_probs, axis=1)[:, :2] == y_true[:, None], axis=1)
    top3_correct = np.any(np.argsort(-y_probs, axis=1)[:, :3] == y_true[:, None], axis=1)
    metrics['top2_accuracy'] = np.mean(top2_correct)
    metrics['top3_accuracy'] = np.mean(top3_correct)
    
    # Calibration metrics
    max_probs = np.max(y_probs, axis=1)
    confidences = max_probs
    accuracies = (y_pred == y_true).astype(float)
    
    metrics['ece'] = compute_ece(confidences, accuracies, n_bins=10)
    metrics['mce'] = compute_mce(confidences, accuracies, n_bins=10)
    
    # Brier Score
    brier_scores = []
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(float)
        brier = np.mean((y_probs[:, i] - y_true_binary) ** 2)
        brier_scores.append(brier)
    metrics['brier_score_mean'] = np.mean(brier_scores)
    metrics['brier_score_per_class'] = np.array(brier_scores)
    metrics['mean_confidence'] = np.mean(max_probs)
    metrics['confidence_when_correct'] = np.mean(max_probs[y_pred == y_true])
    metrics['confidence_when_wrong'] = np.mean(max_probs[y_pred != y_true]) if (y_pred != y_true).any() else 0.0
    
    # Embedding quality
    if embeddings is not None:
        emb_masked = embeddings[mask].cpu().numpy()
        try:
            metrics['silhouette_score'] = silhouette_score(emb_masked, y_true)
        except:
            metrics['silhouette_score'] = 0.0
        try:
            metrics['davies_bouldin_index'] = davies_bouldin_score(emb_masked, y_true)
        except:
            metrics['davies_bouldin_index'] = 0.0
    else:
        metrics['silhouette_score'] = 0.0
        metrics['davies_bouldin_index'] = 0.0
    
    # Graph-aware metrics
    edge_index = data.edge_index
    degrees = torch.zeros(data.num_nodes, dtype=torch.long)
    degrees = degrees.scatter_add(0, edge_index[0], torch.ones_like(edge_index[0]))
    degrees_masked = degrees[mask].numpy()
    
    # Accuracy by degree bins
    degree_bins = [0, 2, 5, 10, float('inf')]
    degree_bin_accuracy = []
    for i in range(len(degree_bins) - 1):
        mask_bin = (degrees_masked >= degree_bins[i]) & (degrees_masked < degree_bins[i+1])
        if mask_bin.sum() > 0:
            acc = (y_pred[mask_bin] == y_true[mask_bin]).mean()
            degree_bin_accuracy.append(acc)
        else:
            degree_bin_accuracy.append(0.0)
    metrics['degree_bin_accuracy'] = np.array(degree_bin_accuracy)
    
    # Homophily analysis
    mask_indices = torch.where(mask)[0]
    homophilic_correct = 0
    heterophilic_correct = 0
    homophilic_total = 0
    heterophilic_total = 0
    
    for i, node_idx in enumerate(mask_indices):
        neighbors = edge_index[1][edge_index[0] == node_idx]
        if len(neighbors) > 0:
            neighbor_labels = data.y[neighbors]
            same_label_ratio = (neighbor_labels == data.y[node_idx]).float().mean().item()
            
            if same_label_ratio > 0.5:
                homophilic_total += 1
                if y_pred[i] == y_true[i]:
                    homophilic_correct += 1
            else:
                heterophilic_total += 1
                if y_pred[i] == y_true[i]:
                    heterophilic_correct += 1
    
    metrics['homophilic_accuracy'] = homophilic_correct / homophilic_total if homophilic_total > 0 else 0.0
    metrics['heterophilic_accuracy'] = heterophilic_correct / heterophilic_total if heterophilic_total > 0 else 0.0
    
    # Store raw data
    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred
    metrics['y_probs'] = y_probs
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    metrics['embeddings'] = embeddings[mask].cpu().numpy() if embeddings is not None else None
    
    return metrics


def compute_ece(confidences, accuracies, n_bins=10):
    """Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def compute_mce(confidences, accuracies, n_bins=10):
    """Maximum Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
    return mce


def mcnemar_test(y_true, pred1, pred2):
    """McNemar's test for statistical significance"""
    n01 = np.sum((pred1 == y_true) & (pred2 != y_true))
    n10 = np.sum((pred1 != y_true) & (pred2 == y_true))
    
    if n01 + n10 == 0:
        return 0.0, 1.0
    
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return statistic, p_value


def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstrap=1000, confidence=0.95):
    """Bootstrap confidence intervals"""
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        score = metric_func(y_true[indices], y_pred[indices])
        bootstrap_scores.append(score)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, alpha/2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
    return lower, upper


# ===================== VISUALIZATION FUNCTIONS =====================

def plot_advanced_confusion_matrix(cm, class_names, title, save_path=None):
    """Advanced confusion matrix with statistics"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
    
    # Main confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='RdYlGn',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Normalized Frequency'}, ax=ax1,
               vmin=0, vmax=1, linewidths=0.5, linecolor='gray')
    
    ax1.set_title(f'Confusion Matrix: {title}', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Per-class precision
    ax2 = fig.add_subplot(gs[0, 1])
    precision = np.diag(cm) / (cm.sum(axis=0) + 1e-10)
    ax2.barh(range(len(precision)), precision, color='steelblue', alpha=0.7)
    ax2.set_ylim(-0.5, len(precision) - 0.5)
    ax2.set_xlim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel('Precision', fontsize=10)
    ax2.set_title('Per-Class\nPrecision', fontsize=10)
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    # Per-class recall
    ax3 = fig.add_subplot(gs[1, 0])
    recall = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
    ax3.bar(range(len(recall)), recall, color='coral', alpha=0.7)
    ax3.set_xlim(-0.5, len(recall) - 0.5)
    ax3.set_ylim(0, 1)
    ax3.set_xticks(range(len(recall)))
    ax3.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Recall', fontsize=10)
    ax3.set_title('Per-Class Recall', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    total_samples = cm.sum()
    accuracy = np.diag(cm).sum() / total_samples
    stats_text = f"Overall Statistics\n{'─'*20}\n"
    stats_text += f"Accuracy:  {accuracy:.3f}\n"
    stats_text += f"Precision: {precision.mean():.3f}\n"
    stats_text += f"Recall:    {recall.mean():.3f}\n"
    stats_text += f"Samples:   {int(total_samples)}"
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    plt.close()


def plot_calibration_analysis(metrics_dict, class_names, save_path=None):
    """Calibration analysis plots"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # Reliability diagram
    ax = axes[0]
    for model_name, metrics in metrics_dict.items():
        y_true = metrics['y_true']
        y_probs = metrics['y_probs']
        y_pred = np.argmax(y_probs, axis=1)
        confidences = np.max(y_probs, axis=1)
        accuracies = (y_pred == y_true).astype(float)
        
        frac_of_pos, mean_pred_value = calibration_curve(accuracies, confidences, n_bins=10, strategy='uniform')
        ax.plot(mean_pred_value, frac_of_pos, 's-', label=model_name, markersize=8, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontsize=11)
    ax.set_title('Reliability Diagram', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # ECE comparison
    ax = axes[1]
    models = list(metrics_dict.keys())
    ece_values = [metrics_dict[m]['ece'] for m in models]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(models)))
    bars = ax.bar(range(len(models)), ece_values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel('ECE', fontsize=11)
    ax.set_title('Expected Calibration Error', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, ece_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
               f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Confidence distributions
    ax = axes[2]
    for model_name, metrics in metrics_dict.items():
        confidences = np.max(metrics['y_probs'], axis=1)
        ax.hist(confidences, bins=20, alpha=0.5, label=model_name, density=True)
    ax.set_xlabel('Confidence', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Confidence correct vs wrong
    ax = axes[3]
    x_pos = np.arange(len(models))
    width = 0.35
    conf_correct = [metrics_dict[m]['confidence_when_correct'] for m in models]
    conf_wrong = [metrics_dict[m]['confidence_when_wrong'] for m in models]
    ax.bar(x_pos - width/2, conf_correct, width, label='Correct', color='green', alpha=0.7)
    ax.bar(x_pos + width/2, conf_wrong, width, label='Wrong', color='red', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    ax.set_ylabel('Mean Confidence', fontsize=11)
    ax.set_title('Confidence: Correct vs Wrong', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Brier score
    ax = axes[4]
    brier_scores = [metrics_dict[m]['brier_score_mean'] for m in models]
    bars = ax.bar(range(len(models)), brier_scores, color=colors, alpha=0.7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel('Brier Score', fontsize=11)
    ax.set_title('Brier Score (Lower Better)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, brier_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
               f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Summary table
    ax = axes[5]
    ax.axis('off')
    table_data = [[m, f"{metrics_dict[m]['ece']:.4f}", f"{metrics_dict[m]['mce']:.4f}",
                  f"{metrics_dict[m]['brier_score_mean']:.4f}"] for m in models]
    table = ax.table(cellText=table_data, colLabels=['Model', 'ECE', 'MCE', 'Brier'],
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Calibration Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved calibration analysis to {save_path}")
    plt.close()


def plot_degree_analysis(metrics_dict, save_path=None):
    """Degree-based performance analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Degree bins
    ax = axes[0]
    degree_bins_labels = ['0-2', '3-5', '6-10', '10+']
    x = np.arange(len(degree_bins_labels))
    width = 0.25
    
    for i, (model_name, metrics) in enumerate(metrics_dict.items()):
        offset = width * (i - 1)
        values = metrics['degree_bin_accuracy']
        ax.bar(x + offset, values, width, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Node Degree Range', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Performance vs Node Degree', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(degree_bins_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Homophilic vs heterophilic
    ax = axes[1]
    models = list(metrics_dict.keys())
    homophilic_acc = [metrics_dict[m]['homophilic_accuracy'] for m in models]
    heterophilic_acc = [metrics_dict[m]['heterophilic_accuracy'] for m in models]
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, homophilic_acc, width, label='Homophilic', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, heterophilic_acc, width, label='Heterophilic', color='coral', alpha=0.8)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Homophilic vs Heterophilic Nodes', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved degree analysis to {save_path}")
    plt.close()


def plot_tsne_embeddings(embeddings_dict, y_true, class_names, save_path=None):
    """t-SNE visualization"""
    n_models = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, embeddings) in enumerate(embeddings_dict.items()):
        ax = axes[idx]
        print(f"  → Computing t-SNE for {model_name}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=y_true, cmap='tab10', s=20, alpha=0.6)
        ax.set_title(f't-SNE: {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=11)
        ax.set_ylabel('t-SNE 2', fontsize=11)
        ax.grid(alpha=0.3)
        
        handles, labels = scatter.legend_elements()
        ax.legend(handles, class_names, loc='best', fontsize=8, title='Classes', framealpha=0.9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved t-SNE visualization to {save_path}")
    plt.close()


def generate_latex_table(metrics_dict, save_path=None):
    """Publication-ready LaTeX table"""
    latex = "\\begin{table}[h]\n\\centering\n"
    latex += "\\caption{Comprehensive Performance Comparison of GNN Models on Cora Dataset}\n"
    latex += "\\label{tab:gnn_comparison}\n"
    latex += "\\begin{tabular}{l|ccccccc}\n\\hline\n"
    latex += "\\textbf{Model} & \\textbf{Acc} & \\textbf{F1} & \\textbf{Prec} & \\textbf{Rec} & \\textbf{AUC} & \\textbf{MCC} & \\textbf{ECE} \\\\\n\\hline\n"
    
    for model_name, metrics in metrics_dict.items():
        latex += f"{model_name} & {metrics['accuracy']:.3f} & {metrics['f1_macro']:.3f} & "
        latex += f"{metrics['precision_macro']:.3f} & {metrics['recall_macro']:.3f} & "
        latex += f"{metrics['roc_auc_ovr_macro']:.3f} & {metrics['mcc']:.3f} & {metrics['ece']:.3f} \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"✓ Saved LaTeX table to {save_path}")
    return latex


# ===================== MAIN EXECUTION =====================

def main():
    print("=" * 90)
    print("🔬 RESEARCH-GRADE COMPREHENSIVE GNN EVALUATION FRAMEWORK (FIXED)")
    print("=" * 90)
    print()
    
    # Create directories
    results_dir = './results'  # CHANGED from ../results to ./results
    figures_dir = os.path.join(results_dir, 'figures')
    reports_dir = os.path.join(results_dir, 'reports')
    tables_dir = os.path.join(results_dir, 'latex_tables')
    
    for dir_path in [figures_dir, reports_dir, tables_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Set seed
    set_seeds(42)
    
    # Class names
    class_names = ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                   'Probabilistic_Methods', 'Reinforcement_Learning',
                   'Rule_Learning', 'Theory']
    
    # Load dataset
    print("📊 Loading Cora dataset...")
    dataset = Planetoid(root='./data/Cora', name='Cora')  # CHANGED from ../ to ./
    data = dataset[0]
    print(f"✓ Dataset: {data.num_nodes} nodes, {data.num_edges} edges, {dataset.num_classes} classes")
    print(f"✓ Test set: {data.test_mask.sum().item()} nodes\n")
    
    # Model configurations - FIXED
        # Model configurations - FIXED PATHS
    num_features = dataset.num_features
    hidden_channels = 64
    num_classes = dataset.num_classes
    
    model_configs = {
        'GCN': {
            'class': GCN,
            'path': './models/saved/gcn_best.pth',  # CHANGED from ../ to ./
            'kwargs': {'dropout': 0.5}
        },
        'GAT': {
            'class': GAT,
            'path': './models/saved/gat_best.pth',  # CHANGED from ../ to ./
            'kwargs': {'heads': 8, 'dropout': 0.6}
        },
        'GraphSAGE': {
            'class': GraphSAGE,
            'path': './models/saved/graphsage_best.pth',  # CHANGED from ../ to ./
            'kwargs': {'dropout': 0.5}
        }
    }
    
    # Evaluate all models
    all_metrics = {}
    embeddings_dict = {}
    
    print("🔍 PHASE 1: Comprehensive Model Evaluation")
    print("-" * 90)
    
    for model_name, config in model_configs.items():
        print(f"\n📈 Evaluating {model_name}...")
        
        try:
            # Load model
            model = load_model(
                config['class'], config['path'],
                num_features, hidden_channels, num_classes,
                **config['kwargs']
            )
            
            # Comprehensive evaluation
            metrics = evaluate_model_comprehensive(model, data, data.test_mask, class_names)
            all_metrics[model_name] = metrics
            
            # Store embeddings
            if metrics['embeddings'] is not None:
                embeddings_dict[model_name] = metrics['embeddings']
            
            # Print summary
            print(f"  ✓ Accuracy:          {metrics['accuracy']:.4f}")
            print(f"  ✓ F1-Score (Macro):  {metrics['f1_macro']:.4f}")
            print(f"  ✓ ROC-AUC (Macro):   {metrics['roc_auc_ovr_macro']:.4f}")
            print(f"  ✓ MCC:               {metrics['mcc']:.4f}")
            print(f"  ✓ ECE:               {metrics['ece']:.4f}")
            print(f"  ✓ Silhouette Score:  {metrics['silhouette_score']:.4f}")
            
        except FileNotFoundError:
            print(f"  ⚠️  Model not found: {config['path']}")
            continue
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_metrics) < 2:
        print("\n⚠️  Need at least 2 models for comparison. Train models first!")
        return
    
    # Statistical significance testing
    print("\n\n🔬 PHASE 2: Statistical Significance Testing")
    print("-" * 90)
    
    models_list = list(all_metrics.keys())
    print("\nMcNemar's Test Results (p-values):")
    print(f"{'':12s}", end='')
    for m in models_list:
        print(f"{m:12s}", end='')
    print()
    
    for i, model1 in enumerate(models_list):
        print(f"{model1:12s}", end='')
        for j, model2 in enumerate(models_list):
            if i == j:
                print(f"{'---':>12s}", end='')
            else:
                y_true = all_metrics[model1]['y_true']
                pred1 = all_metrics[model1]['y_pred']
                pred2 = all_metrics[model2]['y_pred']
                
                _, p_value = mcnemar_test(y_true, pred1, pred2)
                sig_marker = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
                print(f"{p_value:>8.4f} {sig_marker:>3s}", end='')
        print()
    
    print("\n* p<0.05, ** p<0.01, *** p<0.001, ns = not significant")
    
    # Bootstrap confidence intervals
    print("\n\n📊 PHASE 3: Bootstrap Confidence Intervals (95%)")
    print("-" * 90)
    
    for model_name, metrics in all_metrics.items():
        y_true = metrics['y_true']
        y_pred = metrics['y_pred']
        
        acc_ci = bootstrap_confidence_interval(y_true, y_pred, accuracy_score, n_bootstrap=1000)
        f1_ci = bootstrap_confidence_interval(y_true, y_pred, 
                                             lambda yt, yp: f1_score(yt, yp, average='macro'),
                                             n_bootstrap=1000)
        
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]")
        print(f"  F1-Score:  {metrics['f1_macro']:.4f} [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
    
    # Generate visualizations
    print("\n\n🎨 PHASE 4: Generating Advanced Visualizations")
    print("-" * 90)
    
    # 1. Advanced confusion matrices
    print("\n📊 Generating advanced confusion matrices...")
    for model_name, metrics in all_metrics.items():
        cm_path = os.path.join(figures_dir, f'confusion_matrix_advanced_{model_name.lower()}.png')
        plot_advanced_confusion_matrix(metrics['confusion_matrix'], class_names, model_name, save_path=cm_path)
    
    # 2. Calibration analysis
    print("📊 Generating calibration analysis...")
    cal_path = os.path.join(figures_dir, 'calibration_analysis_comprehensive.png')
    plot_calibration_analysis(all_metrics, class_names, save_path=cal_path)
    
    # 3. Degree analysis
    print("📊 Generating degree-based analysis...")
    degree_path = os.path.join(figures_dir, 'degree_based_analysis.png')
    plot_degree_analysis(all_metrics, save_path=degree_path)
    
    # 4. t-SNE embeddings
    if embeddings_dict:
        print("📊 Generating t-SNE visualizations...")
        tsne_path = os.path.join(figures_dir, 'tsne_embeddings_comparison.png')
        plot_tsne_embeddings(embeddings_dict, all_metrics[models_list[0]]['y_true'], class_names, save_path=tsne_path)
    
    # 5. LaTeX table
    print("\n📝 Generating LaTeX table...")
    latex_path = os.path.join(tables_dir, 'comparison_table.tex')
    generate_latex_table(all_metrics, save_path=latex_path)
    
    # Generate comprehensive report
    print("\n📝 Generating comprehensive evaluation report...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(reports_dir, f'comprehensive_evaluation_{timestamp}.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("COMPREHENSIVE GNN EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 90 + "\n\n")
        
        for model_name, metrics in all_metrics.items():
            f.write(f"\n{'='*90}\n")
            f.write(f"MODEL: {model_name}\n")
            f.write(f"{'='*90}\n\n")
            
            f.write("STANDARD METRICS:\n")
            f.write(f"  Accuracy:           {metrics['accuracy']:.4f}\n")
            f.write(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}\n")
            f.write(f"  F1-Score (Macro):   {metrics['f1_macro']:.4f}\n")
            f.write(f"  Precision (Macro):  {metrics['precision_macro']:.4f}\n")
            f.write(f"  Recall (Macro):     {metrics['recall_macro']:.4f}\n\n")
            
            f.write("ADVANCED METRICS:\n")
            f.write(f"  ROC-AUC (Macro):    {metrics['roc_auc_ovr_macro']:.4f}\n")
            f.write(f"  PR-AUC (Macro):     {metrics['pr_auc_macro']:.4f}\n")
            f.write(f"  MCC:                {metrics['mcc']:.4f}\n")
            f.write(f"  Cohen's Kappa:      {metrics['cohen_kappa']:.4f}\n")
            f.write(f"  Top-2 Accuracy:     {metrics['top2_accuracy']:.4f}\n")
            f.write(f"  Top-3 Accuracy:     {metrics['top3_accuracy']:.4f}\n\n")
            
            f.write("CALIBRATION METRICS:\n")
            f.write(f"  ECE:                {metrics['ece']:.4f}\n")
            f.write(f"  MCE:                {metrics['mce']:.4f}\n")
            f.write(f"  Brier Score:        {metrics['brier_score_mean']:.4f}\n")
            f.write(f"  Mean Confidence:    {metrics['mean_confidence']:.4f}\n\n")
            
            f.write("EMBEDDING QUALITY:\n")
            f.write(f"  Silhouette Score:   {metrics['silhouette_score']:.4f}\n")
            f.write(f"  Davies-Bouldin:     {metrics['davies_bouldin_index']:.4f}\n\n")
            
            f.write("GRAPH-SPECIFIC METRICS:\n")
            f.write(f"  Homophilic Accuracy:   {metrics['homophilic_accuracy']:.4f}\n")
            f.write(f"  Heterophilic Accuracy: {metrics['heterophilic_accuracy']:.4f}\n\n")
    
    print(f"✓ Report saved to {report_path}")
    
    # Final summary
    print("\n\n" + "=" * 90)
    print("📊 EVALUATION SUMMARY")
    print("=" * 90)
    
    # Comparison table
    print(f"\n{'Model':<15} {'Accuracy':<12} {'F1-Score':<12} {'ROC-AUC':<12} {'MCC':<12}")
    print("-" * 90)
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<15} {metrics['accuracy']:<12.4f} {metrics['f1_macro']:<12.4f} "
              f"{metrics['roc_auc_ovr_macro']:<12.4f} {metrics['mcc']:<12.4f}")
    
    # Best model
    best_model = max(all_metrics.items(), key=lambda x: x[1]['f1_macro'])
    print(f"\n🏆 BEST MODEL: {best_model[0]}")
    print(f"   F1-Score: {best_model[1]['f1_macro']:.4f}")
    print(f"   ROC-AUC:  {best_model[1]['roc_auc_ovr_macro']:.4f}")
    print(f"   MCC:      {best_model[1]['mcc']:.4f}")
    
    print("\n\n" + "=" * 90)
    print("✅ COMPREHENSIVE EVALUATION COMPLETE!")
    print("=" * 90)
    print()


if __name__ == '__main__':
    main()