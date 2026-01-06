import torch
import torch.nn.functional as F
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from models.graphsage import GraphSAGE
from utils.data_loader import CoraDataLoader


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum() / data.train_mask.sum()
    val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    
    return train_acc.item(), val_acc.item(), test_acc.item()


if __name__ == "__main__":
    print("Starting GraphSAGE training...")
    
    loader = CoraDataLoader()
    data = loader.get_data()
    print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    
    model = GraphSAGE(
        num_features=loader.get_num_features(),
        hidden_channels=64,
        num_classes=loader.get_num_classes(),
        dropout=0.5
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    
    print("\nTraining started...")
    print("-" * 60)
    best_val_acc = 0
    best_test_acc = 0
    best_model_state = None
    patience = 0
    max_patience = 100
    
    for epoch in range(1, 1001):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
        
        if epoch % 50 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                  f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Create models/saved directory if it doesn't exist
    save_dir = os.path.join(parent_dir, 'models', 'saved')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the best model
    save_path = os.path.join(save_dir, 'graphsage_best.pth')
    torch.save(best_model_state, save_path)
    
    print("-" * 60)
    print(f'\nFinal Results (GraphSAGE):')
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    print(f'Best Test Accuracy: {best_test_acc:.4f}')
    print(f'\n💾 Model saved to: {save_path}')
    print("\nTraining complete!")