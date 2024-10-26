import sys
import os
# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from modeling import GUTNetConfig, GUTNetForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import logging
from src.data_processing.dataset_builder import preprocess_data  # Updated import statement
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GUTNetTrainer:
    def __init__(self, config, data_path, batch_size=32):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = GUTNetForSequenceClassification(config).to(self.device)
        # Prepare datasets using preprocess_data
        data_processor = preprocess_data(root_data_path=data_path, batch_size=batch_size)
        self.train_loader, self.val_loader, self.test_loader = data_processor.get_dataloader()
        
        # Add this debugging code
        for i, (inputs, labels) in enumerate(self.train_loader):
            print(f"Batch {i} - Input shape: {inputs.shape}, Labels shape: {labels.shape}")
            print(f"Batch {i} - Labels min: {labels.min().item()}, max: {labels.max().item()}")
            print(f"Batch {i} - Unique labels: {torch.unique(labels)}")
            if i == 2:  # Print info for first 3 batches only
                break
        
        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=1e-3)
        # Loss function
        self.criterion = CrossEntropyLoss()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                inputs, labels = batch
                labels = labels.long()
                
                # Debugging information
                print(f"Labels shape: {labels.shape}")
                print(f"Labels min: {labels.min().item()}, max: {labels.max().item()}")
                print(f"Unique labels: {torch.unique(labels)}")
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs, labels=labels)
                
                # Add this debugging information
                print(f"Model output logits shape: {outputs.logits.shape}")
                
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
            # 评估验证集
            val_accuracy = self.evaluate(self.val_loader)
            logger.info(f"Validation Accuracy: {val_accuracy:.4f}")

    def evaluate(self, data_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = [b.to(self.device) for b in batch]
                outputs = self.model(inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        logger.info("\nClassification Report:")
        logger.info(classification_report(all_labels, all_preds))
        
        return accuracy

    def predict(self, test_loader):
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
        
        return np.array(all_preds)

if __name__ == "__main__":
    # Example usage
    data_path = 'data/standardized'
    input_dim = 11  # This is correct as your input shape is [128, 11]
    num_classes = 5  # This is correct, keep it as 5
    hidden_size = 256
    num_hidden_layers = 8
    num_attention_heads = 4
    intermediate_size = 512
    hidden_dropout_prob = 0.2
    attention_probs_dropout_prob = 0.1
    batch_size = 256
    network_type = 'without_attention'


    # Create configuration
    config = GUTNetConfig(
        input_dim=input_dim,
        num_classes=num_classes, 
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        problem_type="single_label_classification",
        log_level="INFO"
    )

    # Initialize trainer
    trainer = GUTNetTrainer(config, data_path, batch_size=batch_size)

    # Print device information
    logger.info(f"Using device: {trainer.device}")
    if trainer.device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Train the model
    trainer.train(num_epochs=10)

    # Evaluate on test set
    test_accuracy = trainer.evaluate(trainer.test_loader)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")


