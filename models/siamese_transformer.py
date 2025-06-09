from models.siamese import SiameseNetwork, TripletLoss
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SiameseTransformerNetwork(SiameseNetwork):
    """A Siamese network with integrated Transformer"""
    def __init__(self, num_layers=2, nhead=4, dim_feedforward=512):
        super().__init__()
        
        # Add Transformer encoder layer after the original CNN
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=256*5*5,  # Inherited from parent CNN output dimension
                nhead=nhead,
                dim_feedforward=dim_feedforward
            ),
            num_layers=num_layers
        )
        
        # Fully connected layer input
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256)
        )

    def forward_once(self, x):
        x = self.cnn(x)  # Reuse parent class CNN
        x = x.view(x.size()[0], -1)  # [batch, 256*5*5]
        
        # Add Transformer processing
        x = x.unsqueeze(1)  # [batch, 1, features]
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        
        x = self.fc(x)
        return x