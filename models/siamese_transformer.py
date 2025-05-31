from models.siamese import SiameseNetwork, TripletLoss
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SiameseTransformerNetwork(SiameseNetwork):
    """融合Transformer的孪生网络"""
    def __init__(self, num_layers=2, nhead=4, dim_feedforward=512):
        super().__init__()
        
        # 在原有CNN后添加Transformer编码层
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=256*5*5,  # 继承自父类CNN输出维度
                nhead=nhead,
                dim_feedforward=dim_feedforward
            ),
            num_layers=num_layers
        )
        
        # 调整全连接层输入维度
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256)
        )

    def forward_once(self, x):
        x = self.cnn(x)  # 复用父类CNN
        x = x.view(x.size()[0], -1)  # [batch, 256*5*5]
        
        # 添加Transformer处理
        x = x.unsqueeze(1)  # [batch, 1, features]
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        
        x = self.fc(x)
        return x