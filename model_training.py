class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        #         self.enc1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)   #For RGB approach
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = nn.ReLU()(self.enc1(x))
        x = self.pool(x)
        x = nn.ReLU()(self.enc2(x))
        x = self.pool(x)
        x = nn.ReLU()(self.enc3(x))
        x = self.pool(x)
        x = nn.ReLU()(self.enc4(x))
        return x  # Extracted features


model = UNet()
model.eval()

# Preprocess and transform images for U-Net input
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

