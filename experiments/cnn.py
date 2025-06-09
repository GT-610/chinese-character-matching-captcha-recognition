import torch
from torch.utils.data import DataLoader
from models.cnn_classifier import CNNCharClassifier, CharDataset, train_cnn, evaluate_cnn
from data_process.load_dataset import load_dataset
from torchvision import transforms
import os
import csv
import matplotlib.pyplot as plt

def cnn_experiment():
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load datasets for training and testing
    train_dataset = CharDataset(load_dataset(data_root='data', train=True), transform=transform)
    test_dataset = CharDataset(load_dataset(data_root='data', train=False), transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize the model and move it to the appropriate device (GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNCharClassifier(num_classes=10, num_positions=4).to(device)

    # Train the model for 10 epochs
    print("Starting training of CNN model...")
    val_accs = []  # Store validation accuracies
    for epoch in range(10):
        train_cnn(model, train_loader, test_loader, epochs=1, device=device)
        val_acc = evaluate_cnn(model, test_loader, device)
        val_accs.append(val_acc)
        print(f'Epoch {epoch+1}/10 | Validation Accuracy: {val_acc:.2%}')

    # Final evaluation of the model
    final_acc = evaluate_cnn(model, test_loader, device, verbose=True)
    print(f"\nFinal test accuracy of CNN model: {final_acc:.2%}")

    # Save results to files and generate plots
    save_results(val_accs, final_acc)

def save_results(val_accs, final_acc):
    os.makedirs('results', exist_ok=True)

    # Save validation accuracies to a CSV file
    with open('results/validation_accuracy.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Validation Accuracy'])
        for i, acc in enumerate(val_accs):
            writer.writerow([i+1, acc])

    # Save final accuracy to a CSV file
    with open('results/final_accuracy.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Final Accuracy'])
        writer.writerow([final_acc])

    # Generate a bar plot of accuracy by position
    position_correct = [0]*4
    for i in range(4):
        position_correct[i] += (preds[:, i] == labels[:, i]).sum().item()
    position_accs = [acc / len(test_loader.dataset) for acc in position_correct]

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 5), position_accs)
    plt.title('Accuracy by Position')
    plt.xlabel('Position')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, 5))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig('figures/position_accuracy_plot.png')
    plt.close()

    print("Results saved to the 'results' folder, and plots saved to the 'figures' folder.")