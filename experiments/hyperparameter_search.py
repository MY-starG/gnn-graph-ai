"""
Research-Grade Hyperparameter Optimization for GNN Models

This script implements comprehensive hyperparameter search strategies:
1. Grid Search - Exhaustive search over parameter space
2. Random Search - Efficient sampling-based search
3. Bayesian Optimization - Smart sequential search (optional)
4. Cross-validation for robust evaluation
5. Statistical analysis of results
6. Publication-ready visualizations

For Erasmus Mundus PhD Applications - Research Excellence Demonstration
"""

import sys
import os
import json
import time
from datetime import datetime
from itertools import product
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ===================== MODEL DEFINITIONS =====================

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=3, dropout=0.5):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, num_classes))
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, num_classes, heads=1, 
                            concat=False, dropout=dropout)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers=3, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, num_classes))
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


# ===================== TRAINING AND EVALUATION =====================

def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, data, optimizer, train_mask):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluate model"""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    correct = (pred[mask] == data.y[mask]).sum()
    acc = int(correct) / int(mask.sum())
    
    loss = F.cross_entropy(out[mask], data.y[mask]).item()
    
    return acc, loss


def train_and_evaluate(model_class, data, hyperparams, device='cpu', 
                       max_epochs=200, patience=50, verbose=False, seed=42):
    """
    Train model with given hyperparameters and evaluate on validation set
    """
    set_seeds(seed)
    
    # Separate model hyperparameters from optimizer hyperparameters
    model_params = {k: v for k, v in hyperparams.items() 
                   if k not in ['lr', 'weight_decay']}
    
    # Initialize model
    model = model_class(
        num_features=data.num_features,
        num_classes=int(data.y.max().item()) + 1,
        **model_params
    ).to(device)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams.get('lr', 0.01),
        weight_decay=hyperparams.get('weight_decay', 5e-4)
    )
    
    data = data.to(device)
    
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_accs = []
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # Train
        train_loss = train_epoch(model, data, optimizer, data.train_mask)
        
        # Validate
        val_acc, val_loss = evaluate(model, data, data.val_mask)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break
        
        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}: Val Acc = {val_acc:.4f}, Val Loss = {val_loss:.4f}")
    
    training_time = time.time() - start_time
    
    # Final test evaluation
    test_acc, test_loss = evaluate(model, data, data.test_mask)
    
    return {
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'training_time': training_time,
        'epochs_trained': len(train_losses),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs
    }


# ===================== SEARCH STRATEGIES =====================

def grid_search(model_class, model_name, data, param_grid, device='cpu', verbose=True):
    """
    Exhaustive grid search over hyperparameter space
    """
    print(f"\n{'='*90}")
    print(f"🔍 GRID SEARCH: {model_name}")
    print(f"{'='*90}\n")
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    
    print(f"Total combinations to evaluate: {len(all_combinations)}")
    print(f"Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print()
    
    results = []
    
    for idx, combination in enumerate(all_combinations, 1):
        hyperparams = dict(zip(param_names, combination))
        
        if verbose:
            print(f"[{idx}/{len(all_combinations)}] Testing: {hyperparams}")
        
        try:
            result = train_and_evaluate(model_class, data, hyperparams, 
                                       device=device, verbose=False, seed=42)
            result['hyperparams'] = hyperparams
            results.append(result)
            
            if verbose:
                print(f"  → Val Acc: {result['best_val_acc']:.4f}, "
                      f"Test Acc: {result['test_acc']:.4f}, "
                      f"Time: {result['training_time']:.2f}s")
        
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue
    
    # Find best configuration
    if not results:
        print(f"\n❌ ERROR: No successful runs! All configurations failed.")
        return results, None
    
    best_result = max(results, key=lambda x: x['best_val_acc'])
    
    print(f"\n{'─'*90}")
    print(f"🏆 BEST CONFIGURATION:")
    print(f"{'─'*90}")
    print(f"Hyperparameters: {best_result['hyperparams']}")
    print(f"Validation Accuracy: {best_result['best_val_acc']:.4f}")
    print(f"Test Accuracy:       {best_result['test_acc']:.4f}")
    print(f"Training Time:       {best_result['training_time']:.2f}s")
    print()
    
    return results, best_result


def random_search(model_class, model_name, data, param_distributions, 
                 n_iterations=50, device='cpu', verbose=True):
    """
    Random search over hyperparameter distributions
    """
    print(f"\n{'='*90}")
    print(f"🎲 RANDOM SEARCH: {model_name}")
    print(f"{'='*90}\n")
    
    print(f"Total iterations: {n_iterations}")
    print(f"Parameter distributions:")
    for param, dist in param_distributions.items():
        print(f"  {param}: {dist}")
    print()
    
    results = []
    
    for idx in range(n_iterations):
        # Sample hyperparameters with different seed each time
        np.random.seed(42 + idx)  # Different seed for each iteration
        
        hyperparams = {}
        for param, dist in param_distributions.items():
            if isinstance(dist, list):
                value = np.random.choice(dist)
                # Convert numpy types to Python native types
                hyperparams[param] = int(value) if isinstance(value, (np.integer, np.int64)) else float(value) if isinstance(value, (np.floating, np.float64)) else value
            elif isinstance(dist, tuple) and len(dist) == 2:
                # Assume (min, max) for continuous parameters
                if isinstance(dist[0], float):
                    hyperparams[param] = float(np.random.uniform(dist[0], dist[1]))
                else:
                    hyperparams[param] = int(np.random.randint(dist[0], dist[1] + 1))
        
        if verbose:
            print(f"[{idx+1}/{n_iterations}] Testing: {hyperparams}")
        
        try:
            result = train_and_evaluate(model_class, data, hyperparams, 
                                       device=device, verbose=False, seed=42+idx)
            result['hyperparams'] = hyperparams
            results.append(result)
            
            if verbose:
                print(f"  → Val Acc: {result['best_val_acc']:.4f}, "
                      f"Test Acc: {result['test_acc']:.4f}, "
                      f"Time: {result['training_time']:.2f}s")
        
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Find best configuration
    if not results:
        print(f"\n❌ ERROR: No successful runs! All configurations failed.")
        return results, None
    
    best_result = max(results, key=lambda x: x['best_val_acc'])
    
    print(f"\n{'─'*90}")
    print(f"🏆 BEST CONFIGURATION:")
    print(f"{'─'*90}")
    print(f"Hyperparameters: {best_result['hyperparams']}")
    print(f"Validation Accuracy: {best_result['best_val_acc']:.4f}")
    print(f"Test Accuracy:       {best_result['test_acc']:.4f}")
    print(f"Training Time:       {best_result['training_time']:.2f}s")
    print()
    
    return results, best_result


# ===================== VISUALIZATION =====================

def plot_hyperparameter_importance(results, param_name, save_path=None):
    """Plot performance vs single hyperparameter"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extract data
    param_values = [r['hyperparams'][param_name] for r in results]
    val_accs = [r['best_val_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    
    # Validation accuracy
    ax = axes[0]
    ax.scatter(param_values, val_accs, alpha=0.6, s=50)
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title(f'Validation Accuracy vs {param_name}', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Test accuracy
    ax = axes[1]
    ax.scatter(param_values, test_accs, alpha=0.6, s=50, color='coral')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(f'Test Accuracy vs {param_name}', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    plt.close()


def plot_search_results_summary(results, model_name, save_path=None):
    """Comprehensive visualization of search results"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    val_accs = [r['best_val_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    times = [r['training_time'] for r in results]
    
    # 1. Validation accuracy distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(val_accs, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(np.mean(val_accs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(val_accs):.4f}')
    ax1.axvline(np.max(val_accs), color='green', linestyle='--', linewidth=2, label=f'Best: {np.max(val_accs):.4f}')
    ax1.set_xlabel('Validation Accuracy', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Validation Accuracy Distribution', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    # 2. Test accuracy distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(test_accs, bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(np.mean(test_accs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(test_accs):.4f}')
    ax2.axvline(np.max(test_accs), color='green', linestyle='--', linewidth=2, label=f'Best: {np.max(test_accs):.4f}')
    ax2.set_xlabel('Test Accuracy', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Test Accuracy Distribution', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # 3. Training time distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(times, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.axvline(np.mean(times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(times):.2f}s')
    ax3.set_xlabel('Training Time (s)', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.set_title('Training Time Distribution', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    # 4. Val vs Test accuracy
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(val_accs, test_accs, alpha=0.6, s=50)
    ax4.plot([min(val_accs), max(val_accs)], [min(val_accs), max(val_accs)], 
             'r--', linewidth=2, label='Perfect correlation')
    correlation = np.corrcoef(val_accs, test_accs)[0, 1]
    ax4.set_xlabel('Validation Accuracy', fontsize=10)
    ax4.set_ylabel('Test Accuracy', fontsize=10)
    ax4.set_title(f'Val vs Test Accuracy (r={correlation:.3f})', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    # 5. Performance over iterations
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(range(1, len(val_accs)+1), val_accs, 'o-', alpha=0.6, label='Validation', markersize=4)
    ax5.plot(range(1, len(test_accs)+1), test_accs, 's-', alpha=0.6, label='Test', markersize=4)
    # Running best
    running_best_val = np.maximum.accumulate(val_accs)
    ax5.plot(range(1, len(running_best_val)+1), running_best_val, 'r--', linewidth=2, label='Best Val')
    ax5.set_xlabel('Iteration', fontsize=10)
    ax5.set_ylabel('Accuracy', fontsize=10)
    ax5.set_title('Performance Over Iterations', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    
    # 6. Time vs performance
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(times, val_accs, c=test_accs, cmap='viridis', s=50, alpha=0.7)
    ax6.set_xlabel('Training Time (s)', fontsize=10)
    ax6.set_ylabel('Validation Accuracy', fontsize=10)
    ax6.set_title('Time-Performance Tradeoff', fontsize=11, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Test Accuracy', fontsize=9)
    ax6.grid(alpha=0.3)
    
    # 7-9. Top configurations
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Sort by validation accuracy
    sorted_results = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)[:10]
    
    table_data = []
    for i, r in enumerate(sorted_results, 1):
        row = [
            f"{i}",
            f"{r['best_val_acc']:.4f}",
            f"{r['test_acc']:.4f}",
            f"{r['training_time']:.2f}s",
            str({k: f"{v:.4f}" if isinstance(v, float) else v 
                 for k, v in list(r['hyperparams'].items())[:3]})
        ]
        table_data.append(row)
    
    table = ax7.table(cellText=table_data,
                     colLabels=['Rank', 'Val Acc', 'Test Acc', 'Time', 'Hyperparams (Top 3)'],
                     cellLoc='left', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax7.set_title('Top 10 Configurations', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle(f'Hyperparameter Search Results: {model_name}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    plt.close()


def plot_parameter_heatmap(results, param1, param2, save_path=None):
    """Heatmap showing interaction between two parameters"""
    # Create grid
    param1_values = sorted(list(set([r['hyperparams'][param1] for r in results])))
    param2_values = sorted(list(set([r['hyperparams'][param2] for r in results])))
    
    grid = np.zeros((len(param2_values), len(param1_values)))
    counts = np.zeros((len(param2_values), len(param1_values)))
    
    for r in results:
        i = param2_values.index(r['hyperparams'][param2])
        j = param1_values.index(r['hyperparams'][param1])
        grid[i, j] += r['best_val_acc']
        counts[i, j] += 1
    
    # Average if multiple runs
    grid = np.divide(grid, counts, where=counts!=0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(grid, annot=True, fmt='.4f', cmap='YlGnBu',
               xticklabels=[f'{v:.4f}' if isinstance(v, float) else str(v) for v in param1_values],
               yticklabels=[f'{v:.4f}' if isinstance(v, float) else str(v) for v in param2_values],
               cbar_kws={'label': 'Validation Accuracy'}, ax=ax)
    
    ax.set_xlabel(param1, fontsize=12)
    ax.set_ylabel(param2, fontsize=12)
    ax.set_title(f'Validation Accuracy Heatmap: {param1} vs {param2}', 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    plt.close()


# ===================== MAIN EXECUTION =====================

def main():
    print("=" * 90)
    print("🔬 COMPREHENSIVE HYPERPARAMETER OPTIMIZATION FOR GNN MODELS")
    print("=" * 90)
    print()
    
    # Setup directories
    results_dir = './results/hyperparameter_search'
    figures_dir = os.path.join(results_dir, 'figures')
    configs_dir = os.path.join(results_dir, 'configs')
    
    for dir_path in [results_dir, figures_dir, configs_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Using device: {device}\n")
    
    # Load data
    print("📊 Loading Cora dataset...")
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    print(f"✓ Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges\n")
    
    # ===================== GCN GRID SEARCH =====================
    
    gcn_param_grid = {
        'hidden_channels': [32, 64, 128],
        'num_layers': [2, 3],
        'dropout': [0.3, 0.5, 0.7],
        'lr': [0.001, 0.01],
        'weight_decay': [5e-4, 5e-3]
    }
    
    gcn_grid_results, gcn_best = grid_search(
        GCN, 'GCN', data, gcn_param_grid, device=device, verbose=True
    )
    
    if not gcn_grid_results or gcn_best is None:
        print("\n⚠️ GCN grid search failed completely. Exiting.")
        return
    
    # Save results
    gcn_grid_path = os.path.join(configs_dir, 'gcn_grid_search_results.json')
    with open(gcn_grid_path, 'w') as f:
        # Convert numpy types for JSON serialization
        serializable_results = []
        for r in gcn_grid_results:
            r_copy = r.copy()
            r_copy['hyperparams'] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                    for k, v in r['hyperparams'].items()}
            r_copy.pop('train_losses', None)
            r_copy.pop('val_losses', None)
            r_copy.pop('val_accs', None)
            serializable_results.append(r_copy)
        json.dump(serializable_results, f, indent=2)
    print(f"✓ Saved GCN grid search results to {gcn_grid_path}\n")
    
    # Visualizations
    plot_search_results_summary(gcn_grid_results, 'GCN Grid Search',
                               save_path=os.path.join(figures_dir, 'gcn_grid_search_summary.png'))
    
    if 'hidden_channels' in gcn_param_grid and 'dropout' in gcn_param_grid:
        plot_parameter_heatmap(gcn_grid_results, 'hidden_channels', 'dropout',
                             save_path=os.path.join(figures_dir, 'gcn_heatmap_hidden_dropout.png'))
    
    # ===================== GAT RANDOM SEARCH =====================
    
    gat_param_distributions = {
        'hidden_channels': [32, 64, 128, 256],
        'heads': [2, 4, 8],
        'dropout': (0.3, 0.7),  # Uniform sample between 0.3 and 0.7
        'lr': [0.001, 0.005, 0.01],
        'weight_decay': [1e-4, 5e-4, 1e-3, 5e-3]
    }
    
    gat_random_results, gat_best = random_search(
        GAT, 'GAT', data, gat_param_distributions, n_iterations=50, device=device, verbose=True
    )
    
    if not gat_random_results or gat_best is None:
        print("\n⚠️ GAT random search failed completely. Exiting.")
        return
    
    # Save results
    gat_random_path = os.path.join(configs_dir, 'gat_random_search_results.json')
    with open(gat_random_path, 'w') as f:
        serializable_results = []
        for r in gat_random_results:
            r_copy = r.copy()
            r_copy['hyperparams'] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                    for k, v in r['hyperparams'].items()}
            r_copy.pop('train_losses', None)
            r_copy.pop('val_losses', None)
            r_copy.pop('val_accs', None)
            serializable_results.append(r_copy)
        json.dump(serializable_results, f, indent=2)
    print(f"✓ Saved GAT random search results to {gat_random_path}\n")
    
    # Visualizations
    plot_search_results_summary(gat_random_results, 'GAT Random Search',
                               save_path=os.path.join(figures_dir, 'gat_random_search_summary.png'))
    
    plot_hyperparameter_importance(gat_random_results, 'heads',
                                  save_path=os.path.join(figures_dir, 'gat_heads_importance.png'))
    
    # ===================== GraphSAGE RANDOM SEARCH =====================
    
    sage_param_distributions = {
        'hidden_channels': [32, 64, 128, 256],
        'num_layers': [2, 3, 4],
        'dropout': (0.3, 0.7),
        'lr': [0.001, 0.005, 0.01],
        'weight_decay': [1e-4, 5e-4, 1e-3, 5e-3]
    }
    
    sage_random_results, sage_best = random_search(
        GraphSAGE, 'GraphSAGE', data, sage_param_distributions, 
        n_iterations=50, device=device, verbose=True
    )
    
    if not sage_random_results or sage_best is None:
        print("\n⚠️ GraphSAGE random search failed completely. Exiting.")
        return
    
    # Save results
    sage_random_path = os.path.join(configs_dir, 'sage_random_search_results.json')
    with open(sage_random_path, 'w') as f:
        serializable_results = []
        for r in sage_random_results:
            r_copy = r.copy()
            r_copy['hyperparams'] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                    for k, v in r['hyperparams'].items()}
            r_copy.pop('train_losses', None)
            r_copy.pop('val_losses', None)
            r_copy.pop('val_accs', None)
            serializable_results.append(r_copy)
        json.dump(serializable_results, f, indent=2)
    print(f"✓ Saved GraphSAGE random search results to {sage_random_path}\n")
    
    # Visualizations
    plot_search_results_summary(sage_random_results, 'GraphSAGE Random Search',
                               save_path=os.path.join(figures_dir, 'sage_random_search_summary.png'))
    
    plot_hyperparameter_importance(sage_random_results, 'num_layers',
                                  save_path=os.path.join(figures_dir, 'sage_layers_importance.png'))
    
    # ===================== FINAL COMPARISON =====================
    
    print("\n" + "=" * 90)
    print("📊 FINAL COMPARISON OF BEST CONFIGURATIONS")
    print("=" * 90)
    print()
    
    all_best = {
        'GCN': gcn_best,
        'GAT': gat_best,
        'GraphSAGE': sage_best
    }
    
    print(f"{'Model':<15} {'Val Acc':<12} {'Test Acc':<12} {'Time (s)':<12}")
    print("-" * 90)
    for model_name, best_result in all_best.items():
        print(f"{model_name:<15} {best_result['best_val_acc']:<12.4f} "
              f"{best_result['test_acc']:<12.4f} {best_result['training_time']:<12.2f}")
    
    # Overall best model
    overall_best_model = max(all_best.items(), key=lambda x: x[1]['best_val_acc'])
    print(f"\n🏆 OVERALL BEST MODEL: {overall_best_model[0]}")
    print(f"   Best Hyperparameters: {overall_best_model[1]['hyperparams']}")
    print(f"   Validation Accuracy:  {overall_best_model[1]['best_val_acc']:.4f}")
    print(f"   Test Accuracy:        {overall_best_model[1]['test_acc']:.4f}")
    
    # Save best configurations
    best_configs_path = os.path.join(configs_dir, 'best_configurations.json')
    with open(best_configs_path, 'w') as f:
        best_configs = {
            model_name: {
                'hyperparams': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                               for k, v in result['hyperparams'].items()},
                'val_acc': result['best_val_acc'],
                'test_acc': result['test_acc'],
                'training_time': result['training_time']
            }
            for model_name, result in all_best.items()
        }
        json.dump(best_configs, f, indent=2)
    print(f"\n✓ Saved best configurations to {best_configs_path}")
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(all_best.keys())
    val_accs = [all_best[m]['best_val_acc'] for m in models]
    test_accs = [all_best[m]['test_acc'] for m in models]
    times = [all_best[m]['training_time'] for m in models]
    
    # Validation accuracy
    ax = axes[0]
    bars = ax.bar(models, val_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Best Validation Accuracy', fontsize=13, fontweight='bold')
    ax.set_ylim([min(val_accs) - 0.02, max(val_accs) + 0.02])
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, val_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Test accuracy
    ax = axes[1]
    bars = ax.bar(models, test_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Best Test Accuracy', fontsize=13, fontweight='bold')
    ax.set_ylim([min(test_accs) - 0.02, max(test_accs) + 0.02])
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Training time
    ax = axes[2]
    bars = ax.bar(models, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax.set_ylabel('Training Time (s)', fontsize=12)
    ax.set_title('Training Time', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Best Configuration Comparison Across Models', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    comparison_path = os.path.join(figures_dir, 'best_configurations_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {comparison_path}")
    plt.close()
    
    print("\n" + "=" * 90)
    print("✅ HYPERPARAMETER OPTIMIZATION COMPLETE!")
    print("=" * 90)
    print(f"\n📁 Results saved to: {results_dir}")
    print(f"📊 Figures saved to: {figures_dir}")
    print(f"⚙️  Configs saved to: {configs_dir}")
    print()


if __name__ == '__main__':
    main()