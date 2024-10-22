import pandas as pd
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ResizeWithPadding:
    def __init__(self, output_size, crop_bottom=50, crop_top=20):
        self.output_size = output_size
        self.crop_bottom = crop_bottom
        self.crop_top = crop_top

    def __call__(self, image):
        # Crop the image to get rid of metadata
        width, height = image.size
        image = image.crop((0, self.crop_top, width, height - self.crop_bottom))
        original_size = image.size

        # Resize the image to 1400x350 and store the ratio
        ratio = min(self.output_size[0] / original_size[0], self.output_size[1] / original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        delta_w = self.output_size[0] - new_size[0]
        delta_h = self.output_size[1] - new_size[1]
        padding = (0, 0, delta_w, delta_h)
        
        # Pad the image with 0 and return alongside the scale ratio
        return ImageOps.expand(image, padding, fill=0), ratio

class IonogramDataset(Dataset):
    def __init__(self, dataframe):
        # Reset index with drop=True gets rid of gaps in the case of any previous filtering
        self.dataframe = dataframe.reset_index(drop=True)
        self.resize = ResizeWithPadding((1400,350))
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe['path'][idx]
        ionogram_present = self.dataframe['ionogram_present'][idx]
        top_point = self.dataframe['top_point'][idx]

        image = Image.open(image_path).convert('L')
        
        image, ratio = self.resize(image)

        image = transforms.ToTensor()(image)

        labels = {
            "image_path": image_path,
            "ionogram_present": torch.tensor(ionogram_present, dtype=torch.float32),
            # Multiply coordinates by the scaling ratio so they make sense
            "top_point": top_point * ratio,
            "ratio": ratio,
        }
        return image, labels

def convert_grid_points(df, column_name):
    # Regex to extract coordinates from string in the form (x,y)
    df[['x', 'y']] = df[column_name].str.extract(r'\((.*),(.*)\)')

    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')

    sentinel = torch.tensor([-1, -1])

    # Find all instances without a number and replace with a sentinel value
    return df.apply(lambda row: torch.tensor([row['x'], row['y']]) if pd.notnull(row['x']) and pd.notnull(row['y']) else sentinel, axis=1)

def load_data(path = 'L:\DATA\ISIS\Phase 3 - QA &Microapp& Media\labeled_data\combined_observer_results - Copy.csv', 
              batch_size=32, val_size = 0.15, test_size=0.1, random_state_val=42):
    df = pd.read_csv(path)

    # Filter to confidence 2 or 3 samples
    #df = df[round(df['confidence'], 0).isin([2,3])].copy() 

    # Convert class labels to numbers
    df['ionogram_present'] = df['ionogram_present'].map({'Yes': 1, 'No': 0})

    # Convert coordinate labels to tensors
    df['top_point'] = convert_grid_points(df, 'grid_point')

    # Split off test_df with a FIXED random value. This will be a consistent subset across all instances/iterations. Only to be evaluated on once many model variations are trained.
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Split off a validation set from the training set.
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state_val)

    train_dataset = IonogramDataset(train_df)
    val_dataset = IonogramDataset(val_df)
    test_dataset = IonogramDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def visualize(original_path, processed_image):
    original_image = Image.open(original_path).convert("L")

    processed_image_pil = transforms.ToPILImage()(processed_image[0])

    fig, axes = plt.subplots(1,2,figsize=(15,5))

    axes[0].imshow(original_image, cmap='gray')
    axes[1].imshow(processed_image_pil, cmap='gray')

    plt.show()

if __name__ == "__main__":
    path = 'L:\DATA\ISIS\Phase 3 - QA &Microapp& Media\labeled_data\combined_observer_results - Copy.csv'
    train_loader, val_loader, test_loader = load_data(path, batch_size=5)

    print(len(train_loader), len(val_loader), len(test_loader))

    for images, labels in train_loader:
        for i in range(min(len(images), 5)):
            print(labels["image_path"][i])
            print(labels["ionogram_present"][i])
            print(labels["top_point"][i])
            #visualize(labels["image_path"][i], images[i])
        exit()
