# ... existing imports ...
from torch.utils.data import DataLoader
from torchvision import transforms
from data_process.load_dataset import load_dataset

def siamese_transformer_experiment(force_retrain=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/saved/siamese_transformer_model.pth'
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((45, 45)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load the pre-trained model if available
    if not force_retrain and os.path.exists(model_path):
        print("Loading pre-trained Transformer model...")
        net = torch.load(model_path, map_location=device).to(device)
    else:
        print("Training a new Transformer model...")
        base_train = load_dataset(train=True)
        train_dataset = SiameseDataset(base_train, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Training loop
        net = SiameseTransformerNetwork().to(device)
        criterion = TripletLoss(margin=1.0)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
        
        for epoch in range(20):
            running_loss = 0.0
            for i, (char_imgs, pos_imgs, neg_imgs) in enumerate(train_loader):
                char_imgs = char_imgs.to(device)
                pos_imgs = pos_imgs.to(device)
                neg_imgs = neg_imgs.to(device)
                
                optimizer.zero_grad()
                anchor = net.forward_once(char_imgs)
                positive = net.forward_once(pos_imgs)
                negative = net.forward_once(neg_imgs)
                total_loss = criterion(anchor, positive, negative)
                
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()
                
            print(f'Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}')
        
        # Save the trained model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(net, model_path)
        print(f"Transformer model saved to {model_path}")

    # Evaluation
    base_test = load_dataset(train=False)
    test_dataset = SiameseDataset(base_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    evaluate_siamese(net, test_loader, device)
    
    return net