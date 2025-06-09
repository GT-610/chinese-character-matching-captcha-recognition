import torch
from torch.utils.data import DataLoader
from models.siamese import SiameseNetwork, TripletLoss
from data_process.load_dataset import load_dataset
from torchvision import transforms
from models.siamese import SiameseDataset
import torch.nn.functional as F
import os
import cv2

def siamese_experiment(force_retrain=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/saved/siamese_model.pth'
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((45, 45)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Check if the model exists and load it if it does
    if not force_retrain and os.path.exists(model_path):
        print("Loading pre-trained model...")
        # Safe load
        with torch.serialization.safe_globals([SiameseNetwork]):
            net = torch.load(
                model_path,
                weights_only=False,
                map_location=device
            ).to(device)
    else:
        print("Training a new model...")
        # Load and package the dataset
        base_train = load_dataset(train=True)
        train_dataset = SiameseDataset(base_train, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize the model
        net = SiameseNetwork().to(device)
        criterion = TripletLoss(margin=1.0)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
        
        # Train loop
        for epoch in range(20):
            running_loss = 0.0
            for i, (char_imgs, pos_imgs, neg_imgs) in enumerate(train_loader):
                char_imgs = char_imgs.to(device)
                pos_imgs = pos_imgs.to(device)
                neg_imgs = neg_imgs.to(device)

                # Forward
                optimizer.zero_grad()
                
                # Triplet loss
                anchor = net.forward_once(char_imgs)
                positive = net.forward_once(pos_imgs)
                negative = net.forward_once(neg_imgs)
                total_loss = criterion(anchor, positive, negative)
                
                total_loss.backward()
                optimizer.step()
                
                running_loss += total_loss.item()
                
            print(f'Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}')

        # Save trained model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(net, model_path)
        print(f"Model saved to {model_path}")

    # Evaluate
    print("\nEvaluating the model...")  # Start evaluating the model
    base_test = load_dataset(train=False)
    test_dataset = SiameseDataset(base_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    evaluate_siamese(net, test_loader, device)
    
    # Generate prediction visualizations after evaluation
    print("\nGenerating prediction visualizations...")
    visualize_predictions(net, test_loader, device, base_test)
    
    return net

def evaluate_siamese(model, test_loader, device):
    model.eval()
    char_correct = 0
    char_total = 0
    captcha_correct = 0
    captcha_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # Unfold nested dimensions to batch dimension
            char_imgs = batch[0].view(-1, 1, 45, 45).to(device)  # [batch*4, 1, 45, 45]
            pos_imgs = batch[1].view(-1, 1, 45, 45).to(device)   # [batch*4, 1, 45, 45]
            
            # Forward pass with proper dimension structure
            anchor = model.forward_once(char_imgs)
            positive = model.forward_once(pos_imgs)
            
            # Distance calculation and dimension restoration
            distances = F.pairwise_distance(anchor, positive)
            predictions = (distances < 0.5).long()
            
            # Reshape predictions to original captcha structure [batch_size, 4]
            predictions = predictions.view(-1, 4)
            
            # Calculate character accuracy
            char_correct += (predictions == 0).sum().item()
            char_total += predictions.numel()
            
            # Calculate full captcha accuracy
            captcha_pred = (predictions.sum(dim=1) == 0)
            captcha_correct += captcha_pred.sum().item()
            captcha_total += captcha_pred.size(0)

    print(f'单字验证准确率: {char_correct/char_total:.4f}')
    print(f'完整验证码准确率: {captcha_correct/captcha_total:.4f}')
    return (char_correct/char_total, captcha_correct/captcha_total)

# Visualize predictions
def visualize_predictions(model, loader, device, base_dataset, num_samples=5):
    import matplotlib.pyplot as plt
    import numpy as np
    
    model.eval()
    indices = np.random.choice(len(base_dataset), num_samples, replace=False)
    
    plt.figure(figsize=(24, 4*num_samples))
    for plot_idx, sample_idx in enumerate(indices):
        sample = base_dataset[sample_idx]
        sample_id = sample['id']
        true_label = sample['label']
        
        # Load original captcha image
        original_img = cv2.imread(os.path.join(sample['captcha_path'], f"{sample_id}.jpg"))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Plot original captcha
        ax = plt.subplot(num_samples, 5, plot_idx*5 + 1)
        plt.imshow(original_img)
        plt.title(f"Sample ID: {sample_id}\nTrue Label: {true_label}", fontsize=10)
        plt.axis('off')

        # Plot predictions for four characters
        with torch.no_grad():
            inputs = [loader.dataset[sample_idx*4 + i] for i in range(4)]
            char_imgs = torch.stack([x[0] for x in inputs]).to(device)
            pos_imgs = torch.stack([x[1] for x in inputs]).to(device)
            
            anchor = model.forward_once(char_imgs)
            positive = model.forward_once(pos_imgs)
            distances = F.pairwise_distance(anchor, positive)
            predictions = (distances < 0.5).cpu().numpy()

        for char_idx in range(4):
            ax = plt.subplot(num_samples, 5, plot_idx*5 + char_idx + 2)
            
            # Get candidate character image
            candidate_img = inputs[char_idx][1].cpu().numpy().squeeze()
            candidate_img = (candidate_img * 0.5 + 0.5) * 255
            
            plt.imshow(candidate_img)
            
            color = 'red' if predictions[char_idx] else 'green'
            plt.title(f"Char {char_idx+1}\ndist={distances[char_idx]:.2f}",
                     color=color, fontsize=9)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'figures/siamese_predictions.png')
    plt.close()
    print(f"Saved results to figures/siamese_predictions.png")
