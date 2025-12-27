import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# U-NET model, framework from Aladdin Persson on YouTube
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) 
        )
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__ (self, in_channels=8, out_channels = 7, features = [64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)

        # Down Sampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottle_neck= DoubleConv(features[-1], features[-1]*2)
        # Up sampling
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride = 2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.final_cov = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            con_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](con_skip)
        return self.final_cov(x)