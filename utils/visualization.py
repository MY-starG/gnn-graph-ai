"""
Advanced Visualization Utilities for GNN Research

Provides comprehensive visualization tools including:
1. Training curves with validation tracking
2. Learning rate schedules and optimization paths
3. Model comparison charts (accuracy, loss, metrics)
4. Performance heatmaps and statistical plots
5. Publication-quality figure generation
6. Multi-model comparative analysis

For Erasmus Mundus PhD Applications - Research Excellence Demonstration
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from datetime import datetime
import json


class GNNVisualizationTools:
    """Comprehensive visualization toolkit for GNN training and evaluation"""
    
    def __init__(self, output_dir='./results/visualizations'):
        """
        Initialize visualization tools
        
        Args:
            output_dir: Directory to save all visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set publication-quality style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Configure matplotlib for better quality
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 14
        
        print(f"Visualization tools initialized. Output: {output_dir}")
    
    def plot_training_curves(self, history, model_name, save_path=None):
        """
        Plot comprehensive training curves
        
        Args:
            history: Dictionary with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'epochs'
            model_name: Name of the model
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = history.get('epochs', range(1, len(history['train_loss']) + 1))
        
        # Loss plot
        ax = axes[0]
        ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3)
        ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{model_name} - Training and Validation Loss')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Mark best epoch
        best_epoch = np.argmin(history['val_loss'])
        ax.axvline(x=epochs[best_epoch], color='green', linestyle='--', alpha=0.5, label=f'Best Epoch: {epochs[best_epoch]}')
        ax.legend(loc='best', framealpha=0.9)
        
        # Accuracy plot
        ax = axes[1]
        ax.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=3)
        ax.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_name} - Training and Validation Accuracy')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Mark best epoch
        ax.axvline(x=epochs[best_epoch], color='green', linestyle='--', alpha=0.5, label=f'Best Epoch: {epochs[best_epoch]}')
        ax.legend(loc='best', framealpha=0.9)
        
        plt.suptitle(f'Training Progress - {model_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{model_name.lower()}_training_curves.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
        plt.close()
    
    def plot_learning_rate_schedule(self, lr_history, model_name, save_path=None):
        """
        Plot learning rate schedule over training
        
        Args:
            lr_history: List of learning rates per epoch
            model_name: Name of the model
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        
        epochs = range(1, len(lr_history) + 1)
        ax.plot(epochs, lr_history, 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'{model_name} - Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Add annotations for significant changes
        lr_array = np.array(lr_history)
        changes = np.where(np.abs(np.diff(lr_array)) > 0.0001)[0]
        for change_idx in changes[:5]:  # Show first 5 significant changes
            ax.annotate(f'LR: {lr_history[change_idx+1]:.6f}',
                       xy=(change_idx+2, lr_history[change_idx+1]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{model_name.lower()}_lr_schedule.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved learning rate schedule to {save_path}")
        plt.close()
    
    def plot_model_comparison(self, results_dict, metric='accuracy', save_path=None):
        """
        Create comparison bar chart for multiple models
        
        Args:
            results_dict: Dictionary {model_name: {'train': value, 'val': value, 'test': value}}
            metric: Name of the metric being compared
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(results_dict.keys())
        train_scores = [results_dict[m]['train'] for m in models]
        val_scores = [results_dict[m]['val'] for m in models]
        test_scores = [results_dict[m]['test'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        bars1 = ax.bar(x - width, train_scores, width, label='Train', alpha=0.8, color='skyblue', edgecolor='black')
        bars2 = ax.bar(x, val_scores, width, label='Validation', alpha=0.8, color='lightgreen', edgecolor='black')
        bars3 = ax.bar(x + width, test_scores, width, label='Test', alpha=0.8, color='salmon', edgecolor='black')
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel(metric.capitalize(), fontweight='bold')
        ax.set_title(f'Model Comparison - {metric.capitalize()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'model_comparison_{metric}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison to {save_path}")
        plt.close()
    
    def plot_metrics_heatmap(self, metrics_df, save_path=None):
        """
        Create heatmap of multiple metrics across models
        
        Args:
            metrics_df: DataFrame with models as rows and metrics as columns
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Normalize metrics for better visualization
        metrics_normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())
        
        sns.heatmap(metrics_normalized, annot=metrics_df.values, fmt='.3f', 
                   cmap='RdYlGn', cbar_kws={'label': 'Normalized Score'},
                   linewidths=0.5, linecolor='gray', ax=ax,
                   vmin=0, vmax=1)
        
        ax.set_title('Model Performance Heatmap', fontweight='bold', pad=20)
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Models', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'metrics_heatmap.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved metrics heatmap to {save_path}")
        plt.close()
    
    def plot_convergence_analysis(self, histories_dict, save_path=None):
        """
        Analyze convergence behavior of multiple models
        
        Args:
            histories_dict: Dictionary {model_name: history_dict}
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Validation loss convergence
        ax = axes[0, 0]
        for model_name, history in histories_dict.items():
            epochs = range(1, len(history['val_loss']) + 1)
            ax.plot(epochs, history['val_loss'], linewidth=2, label=model_name, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Validation Loss Convergence')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Validation accuracy convergence
        ax = axes[0, 1]
        for model_name, history in histories_dict.items():
            epochs = range(1, len(history['val_acc']) + 1)
            ax.plot(epochs, history['val_acc'], linewidth=2, label=model_name, marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Validation Accuracy Convergence')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Training loss smoothness
        ax = axes[1, 0]
        for model_name, history in histories_dict.items():
            train_loss = np.array(history['train_loss'])
            # Calculate moving average
            window = 5
            if len(train_loss) >= window:
                smoothed = np.convolve(train_loss, np.ones(window)/window, mode='valid')
                epochs_smooth = range(1, len(smoothed) + 1)
                ax.plot(epochs_smooth, smoothed, linewidth=2, label=model_name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Smoothed Training Loss')
        ax.set_title('Training Loss Smoothness (Moving Avg)')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Overfitting analysis (train-val gap)
        ax = axes[1, 1]
        for model_name, history in histories_dict.items():
            train_acc = np.array(history['train_acc'])
            val_acc = np.array(history['val_acc'])
            gap = train_acc - val_acc
            epochs = range(1, len(gap) + 1)
            ax.plot(epochs, gap, linewidth=2, label=model_name, marker='d', markersize=3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train-Val Accuracy Gap')
        ax.set_title('Overfitting Analysis')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Convergence Analysis', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'convergence_analysis.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence analysis to {save_path}")
        plt.close()
    
    def plot_performance_radar(self, metrics_dict, save_path=None):
        """
        Create radar chart comparing models across multiple metrics
        
        Args:
            metrics_dict: Dictionary {model_name: {metric: value}}
            save_path: Path to save figure
        """
        from math import pi
        
        # Get all metrics
        all_metrics = list(next(iter(metrics_dict.values())).keys())
        num_metrics = len(all_metrics)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Compute angle for each metric
        angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
        angles += angles[:1]
        
        # Plot each model
        colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_dict)))
        
        for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
            values = [metrics[m] for m in all_metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_metrics, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9)
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9)
        plt.title('Multi-Metric Performance Comparison', size=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'performance_radar.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved radar chart to {save_path}")
        plt.close()
    
    def create_summary_dashboard(self, histories_dict, final_metrics_dict, save_path=None):
        """
        Create comprehensive dashboard with all key visualizations
        
        Args:
            histories_dict: Dictionary of training histories
            final_metrics_dict: Dictionary of final metrics for each model
            save_path: Path to save figure
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Final accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        models = list(final_metrics_dict.keys())
        accuracies = [final_metrics_dict[m]['test_accuracy'] for m in models]
        colors_acc = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
        bars = ax1.bar(models, accuracies, color=colors_acc, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Final Test Accuracy', fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.3f}', ha='center', fontweight='bold')
        
        # 2. Validation loss over time
        ax2 = fig.add_subplot(gs[0, 1])
        for model_name, history in histories_dict.items():
            epochs = range(1, len(history['val_loss']) + 1)
            ax2.plot(epochs, history['val_loss'], linewidth=2, label=model_name, marker='o', markersize=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_title('Validation Loss Progression', fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. F1-Score comparison
        ax3 = fig.add_subplot(gs[0, 2])
        f1_scores = [final_metrics_dict[m]['f1_score'] for m in models]
        bars = ax3.bar(models, f1_scores, color='skyblue', alpha=0.8, edgecolor='black')
        ax3.set_ylabel('F1-Score')
        ax3.set_title('F1-Score Comparison', fontweight='bold')
        ax3.set_ylim([0, 1])
        ax3.grid(axis='y', alpha=0.3)
        for bar, f1 in zip(bars, f1_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{f1:.3f}', ha='center', fontweight='bold')
        
        # 4. Training accuracy over time
        ax4 = fig.add_subplot(gs[1, 0])
        for model_name, history in histories_dict.items():
            epochs = range(1, len(history['train_acc']) + 1)
            ax4.plot(epochs, history['train_acc'], linewidth=2, label=model_name, marker='s', markersize=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Training Accuracy')
        ax4.set_title('Training Accuracy Progression', fontweight='bold')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Convergence speed (epochs to best validation)
        ax5 = fig.add_subplot(gs[1, 1])
        epochs_to_best = []
        for model_name, history in histories_dict.items():
            best_epoch = np.argmin(history['val_loss']) + 1
            epochs_to_best.append(best_epoch)
        bars = ax5.bar(models, epochs_to_best, color='coral', alpha=0.8, edgecolor='black')
        ax5.set_ylabel('Epochs')
        ax5.set_title('Convergence Speed', fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        for bar, epochs in zip(bars, epochs_to_best):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{epochs}', ha='center', fontweight='bold')
        
        # 6. Precision vs Recall
        ax6 = fig.add_subplot(gs[1, 2])
        precisions = [final_metrics_dict[m]['precision'] for m in models]
        recalls = [final_metrics_dict[m]['recall'] for m in models]
        for i, model in enumerate(models):
            ax6.scatter(recalls[i], precisions[i], s=200, alpha=0.7, label=model, edgecolors='black', linewidth=2)
            ax6.annotate(model, (recalls[i], precisions[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9, fontweight='bold')
        ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax6.set_xlabel('Recall')
        ax6.set_ylabel('Precision')
        ax6.set_title('Precision vs Recall', fontweight='bold')
        ax6.set_xlim([0, 1])
        ax6.set_ylim([0, 1])
        ax6.grid(True, alpha=0.3)
        
        # 7. Training time comparison
        ax7 = fig.add_subplot(gs[2, 0])
        if 'training_time' in final_metrics_dict[models[0]]:
            times = [final_metrics_dict[m]['training_time'] for m in models]
            bars = ax7.bar(models, times, color='lightgreen', alpha=0.8, edgecolor='black')
            ax7.set_ylabel('Time (seconds)')
            ax7.set_title('Training Time', fontweight='bold')
            ax7.grid(axis='y', alpha=0.3)
            for bar, time in zip(bars, times):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{time:.1f}s', ha='center', fontweight='bold')
        
        # 8. Best validation accuracy
        ax8 = fig.add_subplot(gs[2, 1])
        best_val_accs = []
        for model_name, history in histories_dict.items():
            best_val_accs.append(max(history['val_acc']))
        bars = ax8.bar(models, best_val_accs, color='mediumpurple', alpha=0.8, edgecolor='black')
        ax8.set_ylabel('Best Validation Accuracy')
        ax8.set_title('Best Validation Performance', fontweight='bold')
        ax8.set_ylim([0, 1])
        ax8.grid(axis='y', alpha=0.3)
        for bar, acc in zip(bars, best_val_accs):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.3f}', ha='center', fontweight='bold')
        
        # 9. Summary statistics table
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        table_data = []
        for model in models:
            table_data.append([
                model,
                f"{final_metrics_dict[model]['test_accuracy']:.3f}",
                f"{final_metrics_dict[model]['f1_score']:.3f}",
                f"{final_metrics_dict[model]['precision']:.3f}",
                f"{final_metrics_dict[model]['recall']:.3f}"
            ])
        
        table = ax9.table(cellText=table_data, 
                         colLabels=['Model', 'Accuracy', 'F1', 'Precision', 'Recall'],
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Highlight best values
        for col in range(1, 5):
            values = [float(table_data[row][col]) for row in range(len(table_data))]
            best_idx = np.argmax(values)
            table[(best_idx + 1, col)].set_facecolor('#FFD700')
        
        plt.suptitle('GNN Models - Comprehensive Performance Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'summary_dashboard.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary dashboard to {save_path}")
        plt.close()


# Example usage function
def create_sample_visualizations():
    """Create sample visualizations with dummy data"""
    
    viz = GNNVisualizationTools()
    
    # Sample training history
    epochs = 100
    history_gcn = {
        'epochs': list(range(1, epochs + 1)),
        'train_loss': np.exp(-np.linspace(0, 3, epochs)) + np.random.normal(0, 0.02, epochs),
        'val_loss': np.exp(-np.linspace(0, 2.5, epochs)) + np.random.normal(0, 0.03, epochs),
        'train_acc': 1 - np.exp(-np.linspace(0, 3, epochs)) + np.random.normal(0, 0.02, epochs),
        'val_acc': 1 - np.exp(-np.linspace(0, 2.5, epochs)) + np.random.normal(0, 0.03, epochs)
    }
    
    # Create visualizations
    viz.plot_training_curves(history_gcn, 'GCN')
    
    print("\nSample visualizations created successfully!")
    print(f"Check the output directory: {viz.output_dir}")


if __name__ == '__main__':
    print("="*80)
    print("Advanced GNN Visualization Utilities")
    print("="*80)
    print("\nThis module provides comprehensive visualization tools for GNN research.")
    print("Import this module in your training scripts to generate publication-quality figures.")
    print("\nExample usage:")
    print("  from utils.visualization import GNNVisualizationTools")
    print("  viz = GNNVisualizationTools()")
    print("  viz.plot_training_curves(history, 'MyModel')")
    print("\nCreating sample visualizations...")
    create_sample_visualizations()