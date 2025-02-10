import os
import cv2
import string
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt  # new
import seaborn as sns  # new
from sklearn.metrics import confusion_matrix, classification_report  # new
from collections import defaultdict  # new import

# Define valid characters: uppercase + lowercase letters (52 classes)
ALL_CHARACTERS = string.ascii_uppercase + string.ascii_lowercase
CHAR2IDX = {char: idx for idx, char in enumerate(ALL_CHARACTERS)}
IDX2CHAR = {idx: char for char, idx in CHAR2IDX.items()}

def segment_characters(pil_img, debug=False):
    """
    Given a PIL image, convert to grayscale, apply thresholding and use contour
    detection (via OpenCV) to extract the bounding boxes for each character.
    Returns list of cropped character images in PIL format sorted left-to-right.
    """
    # Convert PIL image to OpenCV format (RGB -> BGR) and then grayscale
    img = np.array(pil_img)
    if img.shape[-1] == 4:
        # Remove alpha channel if present
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur and threshold
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and extract bounding boxes
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > 50:  # filter small noise
            bboxes.append((x, y, w, h))
    
    # Sort boxes left-to-right based on x coordinate
    bboxes = sorted(bboxes, key=lambda box: box[0])
    
    crops = []
    for bbox in bboxes:
        x, y, w, h = bbox
        crop = pil_img.crop((x, y, x+w, y+h))
        crops.append(crop)
    return crops

class CaptchaCharDataset(Dataset):
    """
    Dataset that takes a root directory with captcha images.
    The image filename is assumed to contain the text (e.g., captcha_ABcd_123.png).
    Each image is segmented into its individual characters using the segmentation function.
    Each sample is one character crop paired with its corresponding label.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # ...existing code...
        all_image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.png')]
        # New change: use only 10000 images picked randomly (if available)
        if len(all_image_paths) > 10000:
            self.image_paths = random.sample(all_image_paths, 10000)
        else:
            self.image_paths = all_image_paths
        # ...existing code...
        self.samples = []  # list of tuples (image_path, char_index, char_position_in_text)
        for img_path in self.image_paths:
            fname = os.path.basename(img_path)
            # Assume filename like "captcha_<text>_<...>.png"
            parts = fname.split('_')
            if len(parts) < 2:
                continue
            # The second token holds the text embedded (adjust parser as needed)
            text = parts[1]
            for pos, char in enumerate(text):
                if char in CHAR2IDX:
                    self.samples.append((img_path, pos, char))
        # For training, we will re-run segmentation per image and then select the crop by order.
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, pos, char = self.samples[idx]
        pil_img = Image.open(img_path).convert('RGB')
        crops = segment_characters(pil_img)
        # If segmentation fails (count mismatch) then pick from a naive split as fallback.
        if len(crops) != len(os.path.basename(img_path).split('_')[1]):
            # fallback: split image equally across width
            w = pil_img.width
            crop_width = w // len(os.path.basename(img_path).split('_')[1])
            crops = [pil_img.crop((i*crop_width, 0, (i+1)*crop_width, pil_img.height)) for i in range(len(os.path.basename(img_path).split('_')[1]))]
        # Ensure pos is within bounds
        if pos >= len(crops):
            crop_img = crops[-1]
        else:
            crop_img = crops[pos]
        # Convert to grayscale for simplicity
        crop_img = crop_img.convert('L')
        if self.transform is not None:
            crop_img = self.transform(crop_img)
        label = CHAR2IDX[char]
        return crop_img, label

class CharacterClassifier(nn.Module):
    def __init__(self, num_classes=52):
        super(CharacterClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # input 1 channel (grayscale)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch} Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f'Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc

def validate(model, device, val_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch} Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f'Val Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc

def evaluate_word_accuracy(model, dataset, transform, device):
    """
    Evaluate full CAPTCHA (word) accuracy by comparing predicted text against 
    ground truth extracted from the filename (assumed format: "captcha_<text>_...png")
    """
    model.eval()
    unique_img_paths = {}
    for img_path, _, _ in dataset.samples:
        unique_img_paths[img_path] = True
    correct = 0
    total = len(unique_img_paths)
    for img_path in tqdm(unique_img_paths.keys(), desc="Evaluating word accuracy"):
        pil_img = Image.open(img_path).convert('RGB')
        crops = segment_characters(pil_img)
        pred_text = ""
        for crop in crops:
            crop = crop.convert('L')
            crop_tensor = transform(crop).unsqueeze(0).to(device)
            output = model(crop_tensor)
            _, pred = torch.max(output, 1)
            pred_text += IDX2CHAR[pred.item()]
        # Extract ground truth from filename (assumes "captcha_<text>_...png")
        fname = os.path.basename(img_path)
        parts = fname.split('_')
        gt_text = parts[1] if len(parts) > 1 else ""
        if pred_text == gt_text:
            correct += 1
        print(f"Image: {fname}, GT: {gt_text}, Pred: {pred_text}")
    word_acc = correct / total
    print(f"Word Accuracy: {word_acc:.4f}")
    return word_acc

def evaluate_word_confusion_matrix(model, dataset, transform, device):
    """
    Evaluate and plot a confusion matrix for full CAPTCHA (word) predictions.
    For each unique image, extract ground truth and predicted words.
    """
    model.eval()
    unique_img_paths = {}
    for img_path, _, _ in dataset.samples:
        unique_img_paths[img_path] = True

    gt_words = []
    pred_words = []
    for img_path in tqdm(unique_img_paths.keys(), desc="Evaluating word confusion matrix"):
        pil_img = Image.open(img_path).convert('RGB')
        crops = segment_characters(pil_img)
        pred_text = ""
        for crop in crops:
            crop = crop.convert('L')
            crop_tensor = transform(crop).unsqueeze(0).to(device)
            output = model(crop_tensor)
            _, pred = torch.max(output, 1)
            pred_text += IDX2CHAR[pred.item()]
        # Ground truth extraction (assumes "captcha_<word>_...png")
        fname = os.path.basename(img_path)
        parts = fname.split('_')
        gt_text = parts[1] if len(parts) > 1 else ""
        gt_words.append(gt_text)
        pred_words.append(pred_text)
    
    # Create label mapping for words (if there are too many unique words, consider filtering)
    unique_words = sorted(list(set(gt_words + pred_words)))
    word2idx = {word: i for i, word in enumerate(unique_words)}
    gt_labels = [word2idx[w] for w in gt_words]
    pred_labels = [word2idx[w] for w in pred_words]
    
    # Compute confusion matrix
    cm = confusion_matrix(gt_labels, pred_labels)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=unique_words, yticklabels=unique_words)
    plt.xlabel('Predicted Word')
    plt.ylabel('Actual Word')
    plt.title('Word-level Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Optionally, print classification report.
    print("Word Classification Report:")
    print(classification_report(gt_labels, pred_labels, target_names=unique_words))

def show_example_word_predictions(model, dataset, transform, device, num_correct=3, num_incorrect=3):
    """
    Show examples of correctly and incorrectly predicted words with segmentation crops.
    """
    model.eval()
    unique_img_paths = {}
    for img_path, _, _ in dataset.samples:
        unique_img_paths[img_path] = True
    correct_examples = []
    incorrect_examples = []
    for img_path in unique_img_paths.keys():
        pil_img = Image.open(img_path).convert('RGB')
        crops = segment_characters(pil_img)
        pred_text = ""
        for crop in crops:
            crop = crop.convert('L')
            crop_tensor = transform(crop).unsqueeze(0).to(device)
            output = model(crop_tensor)
            _, pred = torch.max(output, 1)
            pred_text += IDX2CHAR[pred.item()]
        fname = os.path.basename(img_path)
        parts = fname.split('_')
        gt_text = parts[1] if len(parts) > 1 else ""
        if pred_text == gt_text:
            if len(correct_examples) < num_correct:
                correct_examples.append((img_path, gt_text, pred_text, crops))
        else:
            if len(incorrect_examples) < num_incorrect:
                incorrect_examples.append((img_path, gt_text, pred_text, crops))
        if len(correct_examples) >= num_correct and len(incorrect_examples) >= num_incorrect:
            break

    for category, examples in zip(["Correct Predictions", "Incorrect Predictions"], [correct_examples, incorrect_examples]):
        for img_path, gt_text, pred_text, crops in examples:
            plt.figure(figsize=(12, 4))
            plt.suptitle(f"{category}: GT: {gt_text} | Pred: {pred_text}", fontsize=14)
            ax1 = plt.subplot(1, len(crops)+1, 1)
            orig_img = Image.open(img_path).convert('RGB')
            ax1.imshow(orig_img)
            ax1.set_title("Original")
            ax1.axis('off')
            for i, crop in enumerate(crops):
                ax = plt.subplot(1, len(crops)+1, i+2)
                ax.imshow(crop, cmap='gray')
                ax.set_title(f"Crop {i+1}")
                ax.axis('off')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

def main(args):
    # Updated device selection for Apple M1 (MPS), CUDA, or CPU fallback.
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms: resize to 32x32 and normalize
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # grayscale normalization
    ])
    
    dataset = CaptchaCharDataset(root_dir=args.data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Update DataLoader settings: use 0 workers and disable pin_memory for MPS backend.
    num_workers = 0 if device.type == "mps" else 4
    pin_memory = False if device.type == "mps" else True
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    model = CharacterClassifier(num_classes=len(ALL_CHARACTERS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Lists to store metrics
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    best_val_acc = 0.0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, device, val_loader, criterion, epoch)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print("Saved best model!")
    
    # Load best model and perform inference as before
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    # ...existing demonstration inference code...
    sample_img, sample_label = dataset[0]
    # Inference: segment full CAPTCHA image and predict characters for visualization
    pil_img = Image.open(dataset.samples[0][0]).convert('RGB')
    crops = segment_characters(pil_img)
    pred_text = ""
    for crop in crops:
        crop = crop.convert('L')
        crop_tensor = transform(crop).unsqueeze(0).to(device)
        output = model(crop_tensor)
        _, pred = torch.max(output, 1)
        pred_text += IDX2CHAR[pred.item()]
    print(f"Predicted text for image {dataset.samples[0][0]}: {pred_text}")
    
    # New: Plot training and validation curves
    epochs = range(1, args.epochs+1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # New: Confusion matrix on the full validation set
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating Confusion Matrix"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(ALL_CHARACTERS), yticklabels=list(ALL_CHARACTERS))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # New: Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(ALL_CHARACTERS)))
    
    # New: Evaluate word accuracy on all unique images.
    evaluate_word_accuracy(model, dataset, transform, device)
    
    # New: Plot word-level confusion matrix.
    evaluate_word_confusion_matrix(model, dataset, transform, device)
    
    # New: Show example word predictions with segmentation visualization.
    show_example_word_predictions(model, dataset, transform, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation + CNN for CAPTCHA character recognition")
    parser.add_argument('--data_dir', type=str, default='hard_captcha_dataset', help='Path to CAPTCHA images')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--save_path', type=str, default='best_character_model.pth', help='Path to save the best model')
    args = parser.parse_args()
    main(args)

# To run the script with Python, execute the following command in the terminal:
# python3 task2_segmentation_cnn.py --data_dir hard_captcha_dataset --epochs 20 --batch_size 64 --save_path best_character_model.pth
