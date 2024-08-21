import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from data_loader import load_data

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(59520, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        #print(x.shape)
        x = self.fc1(x)
        return x

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_loader, val_loader, test_loader = load_data()

    model = SimpleCNN().to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Params:', pytorch_total_params)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    best_val_loss = float('Inf')
    earlystop = 0

    for epoch in range(num_epochs):
        if earlystop > 8:
            print(f"Early stopping on epoch {epoch+1}")
            break
        model.train()
        train_loss = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            top_point = labels["top_point"].to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, top_point)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += top_point.size(0)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss/total}")

        model.eval()
        val_loss = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                images = images.to(device)
                top_point = labels["top_point"].to(device)

                outputs = model(images).squeeze()
                loss = criterion(outputs, top_point)

                val_loss += loss.item()
                total += top_point.size(0)

            print(f"Epoch {epoch+1}, Val Loss: {val_loss/total}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                earlystop = 0
                print(f"New best: {val_loss/total} on epoch {epoch+1}")
                torch.save(model.state_dict(), f"checkpoints/Epoch_{epoch+1}_new_best.pth")
            else:
                earlystop += 1
                torch.save(model.state_dict(), f"checkpoints/Epoch_{epoch+1}.pth")
        
if __name__ == "__main__":
    train()