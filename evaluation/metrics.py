import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_model(model, data, mask_type='test'):
    """
    Comprehensive evaluation of a model
    
    Args:
        model: Trained GNN model
        data: PyG data object
        mask_type: 'train', 'val', or 'test'
    
    Returns:
        dict: Dictionary containing all metrics
    """
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
    
    # Get the appropriate mask
    if mask_type == 'train':
        mask = data.train_mask
    elif mask_type == 'val':
        mask = data.val_mask
    else:
        mask = data.test_mask
    
    # Get predictions and true labels
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    return metrics, y_true, y_pred


def print_metrics(metrics, model_name="Model"):
    """Print metrics in a formatted way"""
    print(f"\n{'='*60}")
    print(f"{model_name} EVALUATION METRICS")
    print(f"{'='*60}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"Precision (Micro):  {metrics['precision_micro']:.4f}")
    print(f"Recall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"Recall (Micro):     {metrics['recall_micro']:.4f}")
    print(f"F1-Score (Macro):   {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Micro):   {metrics['f1_micro']:.4f}")
    print(f"{'='*60}\n")


def plot_confusion_matrix(cm, class_names, model_name="Model", save_path=None):
    """
    Plot confusion matrix as heatmap
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        model_name: Name of the model
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_per_class_metrics(metrics, class_names, model_name="Model", save_path=None):
    """
    Plot per-class precision, recall, and F1-score
    
    Args:
        metrics: Dictionary containing per-class metrics
        class_names: List of class names
        model_name: Name of the model
        save_path: Path to save the figure
    """
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, metrics['precision_per_class'], width, label='Precision', alpha=0.8)
    ax.bar(x, metrics['recall_per_class'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, metrics['f1_per_class'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Per-Class Metrics - {model_name}', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics plot saved to {save_path}")
    
    plt.close()


def compare_models_metrics(models_metrics, model_names, save_path=None):
    """
    Create comparison plot for multiple models
    
    Args:
        models_metrics: List of metrics dictionaries
        model_names: List of model names
        save_path: Path to save the figure
    """
    metrics_to_plot = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (metrics, name) in enumerate(zip(models_metrics, model_names)):
        values = [metrics[m] for m in metrics_to_plot]
        ax.bar(x + i * width, values, width, label=name, alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison - All Metrics', fontsize=16, pad=20)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {save_path}")
    
    plt.close()


def save_metrics_to_file(metrics, model_name, class_names, save_path):
    """Save all metrics to a text file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write(f"{model_name} EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-"*60 + "\n")
        f.write(f"Accuracy:           {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (Macro):  {metrics['precision_macro']:.4f}\n")
        f.write(f"Precision (Micro):  {metrics['precision_micro']:.4f}\n")
        f.write(f"Recall (Macro):     {metrics['recall_macro']:.4f}\n")
        f.write(f"Recall (Micro):     {metrics['recall_micro']:.4f}\n")
        f.write(f"F1-Score (Macro):   {metrics['f1_macro']:.4f}\n")
        f.write(f"F1-Score (Micro):   {metrics['f1_micro']:.4f}\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-"*60 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"\nClass: {class_name}\n")
            f.write(f"  Precision: {metrics['precision_per_class'][i]:.4f}\n")
            f.write(f"  Recall:    {metrics['recall_per_class'][i]:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_per_class'][i]:.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"Metrics report saved to {save_path}")


# Cora class names
CORA_CLASS_NAMES = [
    'Case_Based',
    'Genetic_Algorithms',
    'Neural_Networks',
    'Probabilistic_Methods',
    'Reinforcement_Learning',
    'Rule_Learning',
    'Theory'
]