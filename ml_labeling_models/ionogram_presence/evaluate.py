import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from train import SimpleCNN
from data_loader import load_data

def eval():
    train_loader, val_loader, test_loader = load_data(batch_size=1)

    # Load model and checkpoint to cuda if possible, otherwise cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    print(device)

    model_path = "checkpoints/acc_0.9524.pth"

    model.load_state_dict(torch.load(model_path))

    model.eval()

    all_labels = []
    all_predictions = []
    all_predictions_raw = []
    all_images = []

    with torch.no_grad():
        i = 0
        for images, labels in tqdm(val_loader):
            # if i > 2:
            #     break

            # i+=1
            pred_labels = model(images)

            all_labels.extend(labels["ionogram_present"].numpy())
            all_predictions_raw.extend(pred_labels.numpy())
            all_predictions.extend(np.round(pred_labels.numpy(), 0))
            all_images.extend(images.numpy())
            
    images = np.array(all_images)
    labels = np.array(all_labels)
    predictions = np.array(all_predictions)
    predictions_raw = np.array(all_predictions_raw)

    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,0])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    plt.figure()
    for i in range(len(predictions)):   
        # Visualize cases where the predicted label is incorrect
        if predictions[i][0] == labels[i]:
            continue
        img = images[i].transpose(1,2,0)
        plt.imshow(img, cmap='gray')
        plt.title(f"Pred label: {predictions_raw[i][0]}, Actual label: {labels[i]}")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    eval()