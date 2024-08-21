import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np

from train import SimpleCNN
from data_loader import load_data

def eval():
    train_loader, val_loader, test_loader = load_data(batch_size=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    print(device)

    model_path = "checkpoints/Epoch_51_Best.pth"

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()

    all_labels = []
    all_predictions = []
    all_images = []
    all_losses = []
    all_paths = []

    criterion = nn.MSELoss()

    with torch.no_grad():
        i = 0
        for images, labels in tqdm(val_loader):
            images.to(device)

            top_points = labels["top_point"].to(device)

            pred_labels = model(images)

            all_labels.extend(top_points.numpy())
            all_predictions.extend(pred_labels.numpy())
            all_images.extend(images.numpy())
            all_losses.extend([criterion(pred_labels, top_points).item()])

            all_paths.extend(labels["image_path"])
            # if i > 100:
            #     break
            i+=1
    images = np.array(all_images)
    labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    distances = np.sqrt(np.array(all_losses))
    print(np.mean(distances))
    print(np.median(distances))

    for _ in range(30):   
        i = random.randint(0,len(images)-1)
        view_coords(images[i], labels[i], predictions[i], distances[i], all_paths[i])

    return images, labels, predictions, distances

def view_coords(image, label, prediction, distance, path):
    image = image.transpose(1,2,0)
    print(path)
    
    plt.figure(figsize=(14,4))
    plt.imshow(image, cmap='gray')
    plt.scatter(label[0], label[1], color="blue", marker='x')
    plt.scatter(prediction[0], prediction[1], color="red", marker='x')
    plt.title(f"Pred label: {prediction}, Actual label: {label}, Dist: {distance}")
    plt.show()

if __name__ == "__main__":
    eval()