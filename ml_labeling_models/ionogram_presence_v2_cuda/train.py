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
        self.fc1 = nn.Linear(58464, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        #print(x.shape)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
    
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_loader, val_loader, test_loader = load_data()

    model = SimpleCNN().to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Params:', pytorch_total_params)

    criterion = nn.BCELoss()

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
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            ionogram_present = labels["ionogram_present"].to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, ionogram_present)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == ionogram_present).sum().item()
            total += ionogram_present.size(0)

        train_acc = correct / total
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/total}, Train Accuracy: {train_acc}")

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                images = images.to(device)
                ionogram_present = labels["ionogram_present"].to(device)

                outputs = model(images).squeeze()
                loss = criterion(outputs, ionogram_present)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == ionogram_present).sum().item()
                total += ionogram_present.size(0)

            val_acc = correct / total
            print(f"Epoch {epoch+1}, Val Loss: {val_loss/total}: Val Accuracy: {val_acc}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                earlystop = 0
                print(f"New best: {val_loss} on epoch {epoch+1}")
                torch.save(model.state_dict(), f"checkpoints/Epoch_{epoch+1}_new_best.pth")
            else:
                earlystop += 1
                torch.save(model.state_dict(), f"checkpoints/Epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()