import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.images = []
        self.filenames = []

        for filename in os.listdir(image_folder):
            filepath = os.path.join(image_folder, filename)
            if os.path.isfile(filepath):
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.images.append(img)
                    self.filenames.append(filename)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.filenames[idx]

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7)  # Output: (64, 1, 1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output: (1, 28, 28)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_autoencoder(model, dataloader, num_epochs=20, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for img, _ in dataloader:
            img = img.to(device)
            output = model(img)
            loss = criterion(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model

def extract_features(model, dataloader):
    model.eval()
    features = []
    filenames = []
    with torch.no_grad():
        for img, filename in dataloader:
            img = img.to(device)
            encoded = model.encoder(img)
            encoded = encoded.view(encoded.size(0), -1)
            features.append(encoded.cpu().numpy())
            filenames.extend(filename)
    features = np.vstack(features)
    return features, filenames

def save_clustered_images(output_folder, image_folder, filenames, labels):
    for label in np.unique(labels):
        cluster_folder = os.path.join(output_folder, f'cluster_{label}')
        if not os.path.exists(cluster_folder):
            os.makedirs(cluster_folder)
    
    for filename, label in zip(filenames, labels):
        cluster_folder = os.path.join(output_folder, f'cluster_{label}')
        original_path = os.path.join(image_folder, filename)
        new_path = os.path.join(cluster_folder, filename)
        shutil.copy(original_path, new_path)

def main(input_folder, output_folder, num_clusters=10, num_epochs=20, batch_size=32, learning_rate=1e-3):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    
    dataset = ImageDataset(input_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = Autoencoder().to(device)
    model = train_autoencoder(model, dataloader, num_epochs=num_epochs, learning_rate=learning_rate)
    
    features, filenames = extract_features(model, dataloader)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(features)
    
    save_clustered_images(output_folder, input_folder, filenames, labels)
    print(f"Clustering completed with {num_clusters} clusters.")

if __name__ == "__main__":
    input_folder = "dataset_patches"
    output_folder = "output_autoencoder"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(input_folder, output_folder, num_clusters=10, num_epochs=20, batch_size=32, learning_rate=1e-3)