# import sys
# import os
# # # Add the project root directory to Python path
# # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# # sys.path.insert(0, project_root)
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from .modeling import GUTNetConfig, GUTNetForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import logging
from src.data_processing.dataset_builder import preprocess_data, get_one  # Updated import statement
from tqdm import tqdm
import optuna

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GUTNetTrainer:
    def __init__(self, config, data_path, batch_size=32, lr=1e-3):
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
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        # Loss function
        self.criterion = CrossEntropyLoss()

        # Add metrics tracking lists
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train(self, num_epochs, patience):
        best_val_accuracy = 0
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(range(num_epochs), desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                inputs, labels = get_one(self.train_loader)
                labels = labels.long()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                
                total_train_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Calculate training metrics
            avg_train_loss = total_train_loss / len(self.train_loader)
            train_accuracy = train_correct / train_total
            
            # Validation phase
            val_accuracy = self.evaluate(self.val_loader)
            val_loss = self.calculate_loss(self.val_loader)
            
            # Store metrics
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered. Best validation accuracy: {best_val_accuracy:.4f}")
                    break
        
        final_accuracy, final_report = self.generate_evaluation_report(self.test_loader)
        logger.info("\n最终测试集评估结果:")
        logger.info(f"准确率: {final_accuracy:.4f}")
        logger.info("详细分类报告:")
        logger.info(f"\n{final_report}")
        
        return best_val_accuracy

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
        return accuracy

    def generate_evaluation_report(self, data_loader):
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
        report = classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
        
        return accuracy, report

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

    def objective(self, trial):
        # Define hyperparameters to optimize
        config = GUTNetConfig(
            input_dim=11,
            num_classes=5,
            hidden_size=trial.suggest_int('hidden_size', 64, 256),
            num_hidden_layers=trial.suggest_int('num_hidden_layers', 2, 12),
            num_attention_heads=2,
            intermediate_size=trial.suggest_int('intermediate_size', 128, 1024),
            hidden_dropout_prob=trial.suggest_float('hidden_dropout_prob', 0.0, 0.5),
            attention_probs_dropout_prob=trial.suggest_float('attention_probs_dropout_prob', 0.0, 0.5),
            problem_type="single_label_classification",
            log_level="INFO"
        )

        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

        # Reinitialize the model and optimizer with new hyperparameters
        self.model = GUTNetForSequenceClassification(config).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

        # Prepare datasets using preprocess_data
        data_processor = preprocess_data(root_data_path=self.data_path, batch_size=batch_size)
        self.train_loader, self.val_loader, self.test_loader = data_processor.get_dataloader()

        # Train the model
        best_val_accuracy = self.train(num_epochs=20, patience=5)

        return best_val_accuracy

    # Add new method to calculate loss
    def calculate_loss(self, data_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = [b.to(self.device) for b in batch]
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
        
        return total_loss / len(data_loader)

# if __name__ == "__main__":
#     # Example usage
#     data_path = 'data/standardized'
#     input_dim = 11  # This is correct as your input shape is [128, 11]
#     num_classes = 5  # This is correct, keep it as 5
#     hidden_size = 258
#     num_hidden_layers = 6
#     num_attention_heads = 2
#     intermediate_size = 1024
#     hidden_dropout_prob = 0.1
#     attention_probs_dropout_prob = 0.3
#     batch_size = 256
#     network_type = 'without_attention'


#     # Create configuration
#     config = GUTNetConfig(
#         input_dim=input_dim,
#         num_classes=num_classes, 
#         hidden_size=hidden_size,
#         num_hidden_layers=num_hidden_layers,
#         num_attention_heads=num_attention_heads,
#         intermediate_size=intermediate_size,
#         hidden_dropout_prob=hidden_dropout_prob,
#         attention_probs_dropout_prob=attention_probs_dropout_prob,
#         problem_type="single_label_classification",
#         log_level="INFO"
#     )

#     # 初始化训练器
#     trainer = GUTNetTrainer(config, data_path, batch_size=batch_size)

#     # 训练模型
#     trainer.train(num_epochs=50, patience=5)

#     # 加载最佳模型
#     trainer.model.load_state_dict(torch.load('best_model.pth'))

#     # 在测试集上评估
#     test_accuracy = trainer.evaluate(trainer.test_loader)
#     logger.info(f"测试准确率: {test_accuracy:.4f}")

#     # 在验证集上评估
#     val_accuracy = trainer.evaluate(trainer.val_loader)
#     logger.info(f"验证准确率: {val_accuracy:.4f}")

#     # 生成详细的评估报告
#     test_accuracy, test_report = trainer.generate_evaluation_report(trainer.test_loader)
#     val_accuracy, val_report = trainer.generate_evaluation_report(trainer.val_loader)

#     logger.info("\n测试集详细评估报告:")
#     logger.info(f"准确率: {test_accuracy:.4f}")
#     logger.info(f"分类报告:\n{test_report}")

#     logger.info("\n验证集详细评估报告:")
#     logger.info(f"准确率: {val_accuracy:.4f}")
#     logger.info(f"分类报告:\n{val_report}")






