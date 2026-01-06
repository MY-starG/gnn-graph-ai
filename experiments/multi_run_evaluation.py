"""
Multiple Runs Evaluation with Statistical Analysis

This script runs each model multiple times to compute:
1. Mean and standard deviation of metrics
2. Confidence intervals (95%)
3. Statistical significance tests
4. Stability and reproducibility analysis
5. Publication-quality error bar plots

Demonstrates statistical rigor and reproducibility for research publication
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
from scipy import stats
import json

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


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


# ===================== TRAINING AND EVALUATION =====================

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_and_evaluate(model, data, optimizer, epochs=200, early_stopping=50):
    """Train and evaluate model"""
    best_val_acc = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean().item()
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        
        history['train_loss'].append(loss.item())
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
    
    # Compute additional metrics on test set
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        probs = torch.softmax(out, dim=1)
        
        # Per-class metrics
        test_mask = data.test_mask
        y_true = data.y[test_mask].cpu().numpy()
        y_pred = pred[test_mask].cpu().numpy()
        y_probs = probs[test_mask].cpu().numpy()
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return {
        'train_acc': train_acc,
        'val_acc': best_val_acc,
        'test_acc': best_test_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'history': history
    }


# ===================== MULTIPLE RUNS MANAGER =====================

class MultipleRunsEvaluator:
    """Manages multiple runs and statistical analysis"""
    
    def __init__(self, data, num_runs=10, output_dir='./results/multiple_runs'):
        self.data = data
        self.num_runs = num_runs
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {
            'GCN': [],
            'GAT': [],
            'GraphSAGE': []
        }
        
        print(f"Multiple Runs Evaluator initialized")
        print(f"Number of runs per model: {num_runs}")
        print(f"Output directory: {output_dir}")
    
    def run_model_multiple_times(self, model_class, model_name, config):
        """Run a single model multiple times with different seeds"""
        print(f"\nRunning {model_name} for {self.num_runs} iterations...")
        
        run_results = []
        
        for run in range(self.num_runs):
            seed = 42 + run
            set_seed(seed)
            
            print(f"  Run {run+1}/{self.num_runs} (seed={seed})...", end=' ')
            
            model = model_class(**config)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            
            result = train_and_evaluate(model, self.data, optimizer, epochs=200, early_stopping=50)
            result['seed'] = seed
            result['run'] = run + 1
            
            run_results.append(result)
            
            print(f"Test Acc: {result['test_acc']:.4f}, F1: {result['f1_score']:.4f}")
        
        self.results[model_name] = run_results
        return run_results
    
    def compute_statistics(self, model_name):
        """Compute statistical measures for a model"""
        results = self.results[model_name]
        
        if not results:
            return None
        
        metrics = ['test_acc', 'val_acc', 'train_acc', 'precision', 'recall', 'f1_score']
        statistics = {}
        
        for metric in metrics:
            values = [r[metric] for r in results]
            
            statistics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'values': values
            }
        
        return statistics
    
    def compare_models_statistically(self):
        """Perform statistical tests to compare models"""
        print("\nPerforming statistical comparisons...")
        
        model_names = ['GCN', 'GAT', 'GraphSAGE']
        test_accs = {name: [r['test_acc'] for r in self.results[name]] for name in model_names}
        
        comparisons = []
        
        # Pairwise t-tests
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                t_stat, p_value = stats.ttest_ind(test_accs[model1], test_accs[model2])
                
                mean1 = np.mean(test_accs[model1])
                mean2 = np.mean(test_accs[model2])
                
                comparisons.append({
                    'model1': model1,
                    'model2': model2,
                    'mean1': mean1,
                    'mean2': mean2,
                    'difference': mean1 - mean2,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
                
                print(f"  {model1} vs {model2}:")
                print(f"    Mean difference: {mean1 - mean2:.4f}")
                print(f"    p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        
        return comparisons
    
    def plot_results_with_error_bars(self):
        """Create comprehensive plots with error bars"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        model_names = ['GCN', 'GAT', 'GraphSAGE']
        metrics = ['test_acc', 'precision', 'recall', 'f1_score', 'val_acc', 'train_acc']
        metric_labels = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'Val Accuracy', 'Train Accuracy']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx // 3, idx % 3]
            
            means = []
            stds = []
            ci_lowers = []
            ci_uppers = []
            
            for model_name in model_names:
                stats_dict = self.compute_statistics(model_name)
                means.append(stats_dict[metric]['mean'])
                stds.append(stats_dict[metric]['std'])
                ci_lowers.append(stats_dict[metric]['ci_lower'])
                ci_uppers.append(stats_dict[metric]['ci_upper'])
            
            x = np.arange(len(model_names))
            
            # Error bars with confidence intervals
            errors_lower = [m - ci_l for m, ci_l in zip(means, ci_lowers)]
            errors_upper = [ci_u - m for m, ci_u in zip(means, ci_uppers)]
            
            colors = ['steelblue', 'coral', 'mediumseagreen']
            bars = ax.bar(x, means, yerr=[errors_lower, errors_upper], 
                         capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5,
                         error_kw={'linewidth': 2, 'elinewidth': 2})
            
            ax.set_xticks(x)
            ax.set_xticklabels(model_names)
            ax.set_ylabel(label, fontweight='bold')
            ax.set_title(f'{label} (mean +/- 95% CI)', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])
            
            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{mean:.3f}\n±{std:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        plt.suptitle(f'Model Performance Comparison ({self.num_runs} runs per model)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'performance_with_error_bars.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Error bar plot saved to {save_path}")
    
    def plot_distribution_boxplots(self):
        """Create box plots showing distribution of results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        model_names = ['GCN', 'GAT', 'GraphSAGE']
        metrics = ['test_acc', 'precision', 'recall', 'f1_score', 'val_acc', 'train_acc']
        metric_labels = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'Val Accuracy', 'Train Accuracy']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx // 3, idx % 3]
            
            data_to_plot = []
            for model_name in model_names:
                values = [r[metric] for r in self.results[model_name]]
                data_to_plot.append(values)
            
            bp = ax.boxplot(data_to_plot, labels=model_names, patch_artist=True,
                           notch=True, showmeans=True, meanline=True)
            
            # Color the boxes
            colors = ['steelblue', 'coral', 'mediumseagreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(label, fontweight='bold')
            ax.set_title(label, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.suptitle(f'Performance Distribution ({self.num_runs} runs per model)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'distribution_boxplots.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Box plot saved to {save_path}")
    
    def create_results_table(self):
        """Create publication-ready results table"""
        model_names = ['GCN', 'GAT', 'GraphSAGE']
        metrics = ['test_acc', 'precision', 'recall', 'f1_score']
        metric_labels = ['Test Acc', 'Precision', 'Recall', 'F1-Score']
        
        table_data = []
        for model_name in model_names:
            stats_dict = self.compute_statistics(model_name)
            row = [model_name]
            for metric in metrics:
                mean = stats_dict[metric]['mean']
                std = stats_dict[metric]['std']
                row.append(f"{mean:.3f} ± {std:.3f}")
            table_data.append(row)
        
        df = pd.DataFrame(table_data, columns=['Model'] + metric_labels)
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, 'results_table.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results table saved to {csv_path}")
        
        # Create LaTeX table
        latex_table = df.to_latex(index=False, escape=False)
        latex_path = os.path.join(self.output_dir, 'results_table.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to {latex_path}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 4))
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
        
        # Highlight best values
        for col_idx, metric in enumerate(metrics, start=1):
            values = [self.compute_statistics(name)[metric]['mean'] for name in model_names]
            best_row = np.argmax(values)
            table[(best_row + 1, col_idx)].set_facecolor('#FFD700')
        
        plt.title(f'Performance Comparison (mean ± std over {self.num_runs} runs)',
                 fontsize=12, fontweight='bold', pad=20)
        
        table_img_path = os.path.join(self.output_dir, 'results_table.png')
        plt.savefig(table_img_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Table visualization saved to {table_img_path}")
        
        return df
    
    def generate_report(self):
        """Generate comprehensive text report"""
        report_path = os.path.join(self.output_dir, 'statistical_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*90 + "\n")
            f.write("MULTIPLE RUNS EVALUATION - STATISTICAL REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of runs per model: {self.num_runs}\n")
            f.write("="*90 + "\n\n")
            
            for model_name in ['GCN', 'GAT', 'GraphSAGE']:
                stats_dict = self.compute_statistics(model_name)
                
                f.write(f"\n{model_name}\n")
                f.write("-"*90 + "\n")
                
                for metric in ['test_acc', 'precision', 'recall', 'f1_score']:
                    stats = stats_dict[metric]
                    f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
                    f.write(f"  Mean:   {stats['mean']:.4f}\n")
                    f.write(f"  Std:    {stats['std']:.4f}\n")
                    f.write(f"  Min:    {stats['min']:.4f}\n")
                    f.write(f"  Max:    {stats['max']:.4f}\n")
                    f.write(f"  Median: {stats['median']:.4f}\n")
                    f.write(f"  95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]\n")
                
                f.write("\n")
            
            # Statistical comparisons
            f.write("\n" + "="*90 + "\n")
            f.write("STATISTICAL COMPARISONS (t-tests)\n")
            f.write("="*90 + "\n\n")
            
            comparisons = self.compare_models_statistically()
            for comp in comparisons:
                f.write(f"{comp['model1']} vs {comp['model2']}:\n")
                f.write(f"  Mean difference: {comp['difference']:.4f}\n")
                f.write(f"  t-statistic: {comp['t_statistic']:.4f}\n")
                f.write(f"  p-value: {comp['p_value']:.4f}\n")
                f.write(f"  Significant: {'Yes' if comp['significant'] else 'No'}\n\n")
        
        print(f"Statistical report saved to {report_path}")
    
    def save_all_results(self):
        """Save all raw results to JSON"""
        json_path = os.path.join(self.output_dir, 'all_runs_data.json')
        
        # Convert results to JSON-serializable format
        json_results = {}
        for model_name, runs in self.results.items():
            json_results[model_name] = []
            for run in runs:
                run_data = run.copy()
                # Remove history as it's too large
                if 'history' in run_data:
                    del run_data['history']
                json_results[model_name].append(run_data)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"All runs data saved to {json_path}")


# ===================== MAIN EXECUTION =====================

def main():
    print("="*90)
    print("MULTIPLE RUNS EVALUATION WITH STATISTICAL ANALYSIS")
    print("="*90)
    print()
    
    # Load data
    print("Loading Cora dataset...")
    dataset = Planetoid(root='./data/Cora', name='Cora')
    data = dataset[0]
    print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges\n")
    
    num_features = dataset.num_features
    hidden_channels = 64
    num_classes = dataset.num_classes
    
    # Initialize evaluator
    evaluator = MultipleRunsEvaluator(data, num_runs=10)
    
    # Run GCN
    print("\n" + "="*90)
    print("RUNNING GCN")
    print("="*90)
    gcn_config = {
        'num_features': num_features,
        'hidden_channels': hidden_channels,
        'num_classes': num_classes,
        'dropout': 0.5
    }
    evaluator.run_model_multiple_times(GCN, 'GCN', gcn_config)
    
    # Run GAT
    print("\n" + "="*90)
    print("RUNNING GAT")
    print("="*90)
    gat_config = {
        'num_features': num_features,
        'hidden_channels': hidden_channels,
        'num_classes': num_classes,
        'heads': 8,
        'dropout': 0.6
    }
    evaluator.run_model_multiple_times(GAT, 'GAT', gat_config)
    
    # Run GraphSAGE
    print("\n" + "="*90)
    print("RUNNING GRAPHSAGE")
    print("="*90)
    graphsage_config = {
        'num_features': num_features,
        'hidden_channels': hidden_channels,
        'num_classes': num_classes,
        'dropout': 0.5
    }
    evaluator.run_model_multiple_times(GraphSAGE, 'GraphSAGE', graphsage_config)
    
    # Generate all outputs
    print("\n" + "="*90)
    print("GENERATING STATISTICAL ANALYSIS")
    print("="*90)
    
    print("\nCreating visualizations...")
    evaluator.plot_results_with_error_bars()
    evaluator.plot_distribution_boxplots()
    evaluator.create_results_table()
    evaluator.generate_report()
    evaluator.save_all_results()
    
    print("\n" + "="*90)
    print("MULTIPLE RUNS EVALUATION COMPLETE")
    print("="*90)
    print(f"\nAll results saved to: {evaluator.output_dir}")
    print("\nGenerated outputs:")
    print("  - Error bar plots (PNG)")
    print("  - Box plots (PNG)")
    print("  - Results table (CSV, TEX, PNG)")
    print("  - Statistical report (TXT)")
    print("  - Raw data (JSON)")
    print("\nDemonstrates statistical rigor and reproducibility!")
    print()


if __name__ == '__main__':
    main()