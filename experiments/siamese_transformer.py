# ... existing imports ...
from torch.utils.data import DataLoader
from torchvision import transforms
from data_process.load_dataset import load_dataset

def siamese_transformer_experiment(force_retrain=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'models/saved/siamese_transformer_model.pth'
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((45, 45)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 模型加载
    if not force_retrain and os.path.exists(model_path):
        print("加载预训练Transformer模型...")
        net = torch.load(model_path, map_location=device).to(device)
    else:
        print("开始训练新Transformer模型...")
        base_train = load_dataset(train=True)
        train_dataset = SiameseDataset(base_train, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # 训练循环
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
        
        # 模型保存
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(net, model_path)
        print(f"Transformer模型已保存至 {model_path}")

    # 评估
    base_test = load_dataset(train=False)
    test_dataset = SiameseDataset(base_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    evaluate_siamese(net, test_loader, device)
    
    return net