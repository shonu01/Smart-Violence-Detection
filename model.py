import torch
import torch.nn as nn
import torchvision.models as models

class CNNLSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=1, num_classes=2):
        super(CNNLSTM, self).__init__()

        # Pretrained CNN (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # remove FC layer
        self.cnn = nn.Sequential(*modules)
        self.cnn_out_dim = resnet.fc.in_features

        # Freeze CNN (optional, speeds up training)
        for param in self.cnn.parameters():
            param.requires_grad = False

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Final classifier
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, 3, H, W)
        batch_size, seq_len, C, H, W = x.shape
        x = x.view(batch_size * seq_len, C, H, W)

        # CNN features
        with torch.no_grad():  # CNN frozen
            features = self.cnn(x)  # (batch*seq_len, 512, 1, 1)
        features = features.view(batch_size, seq_len, -1)  # (batch, seq_len, 512)

        # LSTM
        lstm_out, _ = self.lstm(features)
        out = lstm_out[:, -1, :]  # take last frame hidden state

        # Classifier
        out = self.fc(out)
        return out
