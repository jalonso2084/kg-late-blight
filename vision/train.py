import os
import pandas as pd
import torch
import timm
import wandb
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- CONFIGURATION ---
# All major settings are here for easy access
CONFIG = {
    "data_path": "data/labels.csv",
    "model_name": "vit_tiny_patch16_224", # A standard tiny Vision Transformer
    "num_classes": 2, # healthy, lateblight
    "batch_size": 32,
    "image_size": 224,
    "learning_rate": 1e-4,
    "epochs": 10,
    "project_name": "kg-late-blight-vision",
    "run_name": "sprint_2_2_finetune",
    "output_model_path": "vision/mosvit_blite.pt",
    "output_log_path": "vision/train_log.md"
}

# --- 1. PYTORCH DATASET CLASS (ROBUST VERSION) ---
# This class teaches PyTorch how to load your specific dataset from the CSV.
class PotatoDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.transform = transform
        # Create a mapping from class name string to integer
        self.class_to_int = {label: i for i, label in enumerate(sorted(self.labels_df['class_label'].unique()))}
        self.int_to_class = {i: label for label, i in self.class_to_int.items()}
        print(f"Class mapping: {self.class_to_int}")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = self.labels_df.iloc[idx, 0]
        label_str = self.labels_df.iloc[idx, 1]
        label_int = self.class_to_int[label_str]

        try:
            # Attempt to open the image
            image = Image.open(img_path).convert("RGB")
        except OSError as e:
            # This 'except' block catches broken images
            print(f"\nWARNING: Skipping corrupted image file: {img_path} - Error: {e}")
            # Return a blank black image and the correct label if an image is broken
            image = Image.new('RGB', (CONFIG["image_size"], CONFIG["image_size"]), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
            
        return image, label_int

# --- 2. THE MAIN TRAINING FUNCTION ---
def train_model():
    # Initialize Weights & Biases for metric tracking
    wandb.init(project=CONFIG["project_name"], name=CONFIG["run_name"], config=CONFIG)
    
    # Check for GPU, fall back to CPU if not available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define data augmentations and transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create dataset instances
    full_dataset = PotatoDataset(csv_file=CONFIG["data_path"])
    
    # Get the splits from the CSV file based on the folder path
    train_df = full_dataset.labels_df[full_dataset.labels_df['image_path'].str.contains('/train/')]
    val_df = full_dataset.labels_df[full_dataset.labels_df['image_path'].str.contains('/val/')]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_df.index)
    val_dataset = torch.utils.data.Subset(full_dataset, val_df.index)
    
    # Assign transforms after splitting
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4) # num_workers can speed up data loading
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
    
    print(f"Training with {len(train_dataset)} images, validating with {len(val_dataset)} images.")

    # Load the pre-trained model
    model = timm.create_model(CONFIG["model_name"], pretrained=True, num_classes=CONFIG["num_classes"])
    model.to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # --- TRAINING LOOP ---
    best_f1 = 0.0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # --- VALIDATION LOOP ---
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {epoch_loss:.4f} | Val F1: {val_f1:.4f}")
        wandb.log({"epoch": epoch, "loss": epoch_loss, "val_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), CONFIG["output_model_path"])
            print(f"  -> New best model saved with F1: {best_f1:.4f}")

    print("\nTraining complete.")
    print(f"Best validation F1 score: {best_f1:.4f}")
    wandb.finish()

# --- 3. SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    train_model()