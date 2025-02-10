import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from datetime import datetime
from captcha_classifier import CaptchaDataset, train_model, evaluate_model, analyze_predictions, save_model
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import random
import json
from visualization_utils import visualize_batch_predictions

class BaselineCaptchaClassifier(nn.Module):
    def __init__(self, vocab_size, max_length):
        super(BaselineCaptchaClassifier, self).__init__()
        
        # Enhanced feed-forward network with more layers
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            # First block - Initial dimension reduction
            nn.Linear(200 * 400, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second block
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Third block
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Fourth block
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Final output layer
            nn.Linear(256, max_length * vocab_size)
        )
        
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Initialize weights for better training
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.flatten(x)
        x = self.mlp(x)
        return x.view(-1, self.max_length, self.vocab_size)

def visualize_predictions(correct_predictions, incorrect_predictions, num_samples=5, save_dir='prediction_analysis'):
    """Visualize and save prediction examples"""
    os.makedirs(save_dir, exist_ok=True)
    
    for pred_type in ['correct', 'incorrect']:
        predictions_list = correct_predictions if pred_type == 'correct' else incorrect_predictions
        if not predictions_list:
            continue
            
        # Take a few samples
        samples = predictions_list[:num_samples]
        
        # Create figure
        fig, axes = plt.subplots(len(samples), 1, figsize=(10, 4*len(samples)))
        if len(samples) == 1:
            axes = [axes]
        
        for idx, pred in enumerate(samples):
            # Convert tensor to image
            img = pred['image'].cpu().numpy()[0]  # Get grayscale channel
            axes[idx].imshow(img, cmap='gray')
            axes[idx].axis('off')
            axes[idx].set_title(f"True: {pred['true']}\nPredicted: {pred['predicted']}\n"
                              f"Confidence: {pred['confidence']:.2f}")
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/baseline_{pred_type}_predictions.png")
        plt.close()

def load_dataset_with_variations(dataset_name, max_variations_per_label=None):
    """Load dataset with control over number of variations per label"""
    images = []
    labels = []
    label_variations = {}
    
    # First pass: group images by label
    for filename in os.listdir(dataset_name):
        if filename.endswith('.png'):
            parts = filename.split('_')
            if len(parts) >= 3 and parts[0] == 'captcha':
                label = parts[1]
                if label not in label_variations:
                    label_variations[label] = []
                label_variations[label].append(os.path.join(dataset_name, filename))
    
    # Second pass: select variations
    for label, variations in label_variations.items():
        if max_variations_per_label is not None:
            # Randomly select up to max_variations_per_label
            selected_variations = random.sample(
                variations, 
                min(max_variations_per_label, len(variations))
            )
        else:
            selected_variations = variations
            
        images.extend(selected_variations)
        labels.extend([label] * len(selected_variations))
    
    print(f"Loaded {len(images)} images for {len(label_variations)} unique labels")
    print(f"Average variations per label: {len(images)/len(label_variations):.1f}")
    
    return images, labels

def main():
    params = {
        'BATCH_SIZE': 32,
        'NUM_EPOCHS': 10,
        'LEARNING_RATE': 0.001,
        'IMAGE_SIZE': (200, 400),
        'TRAIN_SPLIT': 0.7,
        'VAL_SPLIT': 0.15,
        'MAX_VARIATIONS': 20,
        'DATASET_TYPE': 'easy'  # Change to 'hard' for hard dataset
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Select dataset path
    dataset_paths = {
        'easy': '/Users/eshaan/Projects/Precog-Task/Data_generation/easy_variations_dataset',
        'hard': '/Users/eshaan/Projects/Precog-Task/Data_generation/hard_variations_dataset'
    }
    
    dataset_path = dataset_paths[params['DATASET_TYPE']]
    print(f"\nTraining baseline model on {params['DATASET_TYPE']} dataset")
    
    # Load images with variation control
    images, labels = load_dataset_with_variations(
        dataset_path, 
        max_variations_per_label=params['MAX_VARIATIONS']
    )
    
    # Create datasets
    transform = transforms.Compose([
        transforms.Resize(params['IMAGE_SIZE']),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])
    
    dataset = CaptchaDataset(images, labels, transform)
    
    # Split sizes
    total_size = len(dataset)
    train_size = int(params['TRAIN_SPLIT'] * total_size)
    val_size = int(params['VAL_SPLIT'] * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params['BATCH_SIZE'], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=params['BATCH_SIZE']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=params['BATCH_SIZE']
    )
    
    # Initialize model
    model = BaselineCaptchaClassifier(dataset.vocab_size, dataset.max_length).to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device, params['NUM_EPOCHS'])
    
    # Analyze and visualize predictions
    correct_preds, incorrect_preds = analyze_predictions(
        model, test_loader, dataset, device
    )
    
    # Visualize predictions
    visualize_batch_predictions(
        correct_preds,
        incorrect_preds,
        'visualizations',
        params['DATASET_TYPE']
    )
    
    # Print detailed analysis
    print("\nDetailed Error Analysis:")
    print(f"Total test samples: {len(correct_preds) + len(incorrect_preds)}")
    print(f"Correct predictions: {len(correct_preds)}")
    print(f"Incorrect predictions: {len(incorrect_preds)}")
    error_rate = len(incorrect_preds) / (len(correct_preds) + len(incorrect_preds))
    print(f"Error rate: {error_rate:.2%}")
    
    # Analyze error patterns
    error_patterns = {}
    for pred in incorrect_preds:
        true = pred['true']
        predicted = pred['predicted']
        pattern = f"{true} → {predicted}"
        error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
    
    print("\nCommon Error Patterns:")
    for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{pattern}: {count} occurrences")
    
    # Visualize examples
    visualize_predictions(correct_preds, incorrect_preds, 
                        num_samples=5, 
                        save_dir=f'prediction_analysis_{params['DATASET_TYPE']}')
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, dataset, device)
    
    # Save model with simplified path
    save_model(model, f"baseline_{params['DATASET_TYPE']}", metrics)
    
    # Save model and results
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'dataset': dataset_path,
        'parameters': params,
        'metrics': metrics
    }
    
    print(f"\nResults for {dataset_path}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Update result file paths
    results_path = os.path.join('results', f"baseline_{params['DATASET_TYPE']}_results.json")
    detailed_results_path = os.path.join('results', f"baseline_{params['DATASET_TYPE']}_detailed_analysis.json")
    
    # Save results with proper paths
    os.makedirs('results', exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save detailed results
    detailed_results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'dataset': dataset_path,
        'total_samples': len(correct_preds) + len(incorrect_preds),
        'correct_samples': len(correct_preds),
        'incorrect_samples': len(incorrect_preds),
        'error_rate': error_rate,
        'error_patterns': error_patterns,
        'parameters': params,
        'metrics': metrics
    }
    
    with open(detailed_results_path, 'w') as f:
        json.dump(detailed_results, f, indent=4)

if __name__ == "__main__":
    main()
