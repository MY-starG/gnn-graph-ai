"""
Augmentation Effect Evaluation Script

This script evaluates the impact of different graph augmentation strategies
on model performance. Compares training with and without augmentation.

Demonstrates modern techniques for robust graph learning
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
from datetime import datetime

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from utils.graph_augmentation import GraphAugmentor, AUGMENTATION_PRESETS


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


# ===================== TRAINING WITH AUGMENTATION =====================

def train_with_augmentation(model, data, augmentor, optimizer, use_augmentation=True):
    """Train for one epoch with optional augmentation"""
    model.train()
    optimizer.zero_grad()
    
    if use_augmentation:
        # Apply augmentation during training
        data_aug = augmentor.augment(data)
        out = model(data_aug.x, data_aug.edge_index)
        
        # Handle case where nodes might be dropped
        train_mask = data_aug.train_mask if hasattr(data_aug, 'train_mask') else data.train_mask[:data_aug.num_nodes]
        y = data_aug.y if hasattr(data_aug, 'y') else data.y[:data_aug.num_nodes]
    else:
        out = model(data.x, data.edge_index)
        train_mask = data.train_mask
        y = data.y
    
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model, data):
    """Evaluate model"""
    model.eval()
    
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    
    return train_acc, val_acc, test_acc


def train_model(model, data, augmentor=None, epochs=200, lr=0.01, early_stopping=50, use_augmentation=True):
    """Train model with or without augmentation"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': []
    }
    
    for epoch in range(epochs):
        # Training
        if use_augmentation and augmentor is not None:
            loss = train_with_augmentation(model, data, augmentor, optimizer, use_augmentation=True)
        else:
            loss = train_with_augmentation(model, data, None, optimizer, use_augmentation=False)
        
        # Evaluation (always on original graph)
        train_acc, val_acc, test_acc = evaluate(model, data)
        
        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping:
            break
    
    return {
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'final_train_acc': train_acc,
        'history': history,
        'epochs_trained': epoch + 1
    }


# ===================== AUGMENTATION EVALUATION =====================

class AugmentationEvaluator:
    """Evaluate effects of different augmentation strategies"""
    
    def __init__(self, data, output_dir='./results/augmentation_effects'):
        self.data = data
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {}
        
        print("Augmentation Evaluator initialized")
        print(f"Output directory: {output_dir}")
    
    def evaluate_augmentation_strategy(self, strategy_name, augmentation_config, num_runs=5):
        """Evaluate a single augmentation strategy"""
        print(f"\nEvaluating: {strategy_name}")
        print(f"Configuration: {augmentation_config}")
        
        augmentor = GraphAugmentor(augmentation_config) if augmentation_config else None
        use_aug = augmentor is not None
        
        run_results = []
        
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...", end=' ')
            
            # Set seed for reproducibility
            torch.manual_seed(42 + run)
            
            # Create model
            model = GCN(
                num_features=self.data.num_features,
                hidden_channels=64,
                num_classes=self.data.y.max().item() + 1,
                dropout=0.5
            )
            
            # Train
            result = train_model(
                model, self.data, augmentor,
                epochs=200, lr=0.01, early_stopping=30,
                use_augmentation=use_aug
            )
            
            run_results.append(result)
            print(f"Test Acc: {result['best_test_acc']:.4f}")
        
        # Compute statistics
        test_accs = [r['best_test_acc'] for r in run_results]
        val_accs = [r['best_val_acc'] for r in run_results]
        
        self.results[strategy_name] = {
            'runs': run_results,
            'test_acc_mean': np.mean(test_accs),
            'test_acc_std': np.std(test_accs),
            'val_acc_mean': np.mean(val_accs),
            'val_acc_std': np.std(val_accs),
            'config': augmentation_config
        }
        
        print(f"  Mean Test Acc: {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}")
        
        return self.results[strategy_name]
    
    def compare_strategies(self):
        """Compare all evaluated strategies"""
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)
        
        strategies = list(self.results.keys())
        
        print(f"\n{'Strategy':<25} {'Test Acc (mean ± std)':<25} {'Val Acc (mean ± std)':<25}")
        print("-"*80)
        
        for strategy in strategies:
            result = self.results[strategy]
            test_str = f"{result['test_acc_mean']:.4f} ± {result['test_acc_std']:.4f}"
            val_str = f"{result['val_acc_mean']:.4f} ± {result['val_acc_std']:.4f}"
            print(f"{strategy:<25} {test_str:<25} {val_str:<25}")
    
    def plot_comparison(self):
        """Create comparison visualizations"""
        strategies = list(self.results.keys())
        
        # Figure 1: Bar chart with error bars
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Test accuracy
        ax = axes[0]
        means = [self.results[s]['test_acc_mean'] for s in strategies]
        stds = [self.results[s]['test_acc_std'] for s in strategies]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(strategies)))
        bars = ax.bar(range(len(strategies)), means, yerr=stds, capsize=5,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.set_ylabel('Test Accuracy', fontweight='bold')
        ax.set_title('Test Accuracy Comparison', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                   f'{mean:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        # Validation accuracy
        ax = axes[1]
        means = [self.results[s]['val_acc_mean'] for s in strategies]
        stds = [self.results[s]['val_acc_std'] for s in strategies]
        
        bars = ax.bar(range(len(strategies)), means, yerr=stds, capsize=5,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.set_ylabel('Validation Accuracy', fontweight='bold')
        ax.set_title('Validation Accuracy Comparison', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                   f'{mean:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'augmentation_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nComparison plot saved to {save_path}")
    
    def plot_training_curves(self):
        """Plot average training curves for each strategy"""
        strategies = list(self.results.keys())
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
        
        # Validation accuracy over epochs
        ax = axes[0]
        for strategy, color in zip(strategies, colors):
            runs = self.results[strategy]['runs']
            
            # Average over runs
            max_len = max(len(r['history']['val_acc']) for r in runs)
            val_accs = np.zeros((len(runs), max_len))
            
            for i, run in enumerate(runs):
                hist = run['history']['val_acc']
                val_accs[i, :len(hist)] = hist
                if len(hist) < max_len:
                    val_accs[i, len(hist):] = hist[-1]  # Pad with last value
            
            mean_val_acc = val_accs.mean(axis=0)
            std_val_acc = val_accs.std(axis=0)
            epochs = range(1, max_len + 1)
            
            ax.plot(epochs, mean_val_acc, linewidth=2, label=strategy, color=color)
            ax.fill_between(epochs, mean_val_acc - std_val_acc, mean_val_acc + std_val_acc,
                           alpha=0.2, color=color)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Validation Accuracy', fontweight='bold')
        ax.set_title('Validation Accuracy Over Training', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        # Training loss over epochs
        ax = axes[1]
        for strategy, color in zip(strategies, colors):
            runs = self.results[strategy]['runs']
            
            max_len = max(len(r['history']['train_loss']) for r in runs)
            train_losses = np.zeros((len(runs), max_len))
            
            for i, run in enumerate(runs):
                hist = run['history']['train_loss']
                train_losses[i, :len(hist)] = hist
                if len(hist) < max_len:
                    train_losses[i, len(hist):] = hist[-1]
            
            mean_loss = train_losses.mean(axis=0)
            std_loss = train_losses.std(axis=0)
            epochs = range(1, max_len + 1)
            
            ax.plot(epochs, mean_loss, linewidth=2, label=strategy, color=color)
            ax.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                           alpha=0.2, color=color)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Training Loss', fontweight='bold')
        ax.set_title('Training Loss Over Time', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_path}")
    
    def create_results_table(self):
        """Create results summary table"""
        strategies = list(self.results.keys())
        
        table_data = []
        for strategy in strategies:
            result = self.results[strategy]
            table_data.append({
                'Strategy': strategy,
                'Test Acc': f"{result['test_acc_mean']:.4f} ± {result['test_acc_std']:.4f}",
                'Val Acc': f"{result['val_acc_mean']:.4f} ± {result['val_acc_std']:.4f}"
            })
        
        df = pd.DataFrame(table_data)
        
        # Save CSV
        csv_path = os.path.join(self.output_dir, 'results_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults table saved to {csv_path}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, len(strategies)*0.6 + 1))
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best test accuracy
        test_accs = [self.results[s]['test_acc_mean'] for s in strategies]
        best_idx = np.argmax(test_accs)
        table[(best_idx + 1, 1)].set_facecolor('#FFD700')
        
        plt.title('Augmentation Strategy Results', fontsize=12, fontweight='bold', pad=20)
        
        table_path = os.path.join(self.output_dir, 'results_table.png')
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Results table visualization saved to {table_path}")


# ===================== MAIN EXECUTION =====================

def main():
    print("="*90)
    print("GRAPH AUGMENTATION EFFECT EVALUATION")
    print("="*90)
    print()
    
    # Load data
    print("Loading Cora dataset...")
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    print(f"Dataset: {data.num_nodes} nodes, {data.num_edges} edges\n")
    
    # Initialize evaluator
    evaluator = AugmentationEvaluator(data)
    
    # Define strategies to evaluate
    strategies = {
        'No Augmentation': None,
        'Light': AUGMENTATION_PRESETS['light'],
        'Moderate': AUGMENTATION_PRESETS['moderate'],
        'Aggressive': AUGMENTATION_PRESETS['aggressive'],
        'Edge Only': AUGMENTATION_PRESETS['edge_only'],
        'Feature Only': AUGMENTATION_PRESETS['feature_only']
    }
    
    # Evaluate each strategy
    print("="*90)
    print("EVALUATING AUGMENTATION STRATEGIES")
    print("="*90)
    
    for strategy_name, config in strategies.items():
        evaluator.evaluate_augmentation_strategy(strategy_name, config, num_runs=5)
    
    # Compare and visualize
    print("\n" + "="*90)
    print("GENERATING ANALYSIS")
    print("="*90)
    
    evaluator.compare_strategies()
    evaluator.plot_comparison()
    evaluator.plot_training_curves()
    evaluator.create_results_table()
    
    print("\n" + "="*90)
    print("EVALUATION COMPLETE")
    print("="*90)
    print(f"\nAll results saved to: {evaluator.output_dir}")
    print("\nGenerated outputs:")
    print("  - Comparison plots (PNG)")
    print("  - Training curves (PNG)")
    print("  - Results table (CSV + PNG)")
    print("\nDemonstrates modern augmentation techniques for robust learning!")
    print()


if __name__ == '__main__':
    main()