import os
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from typing import Tuple, Generator

# Set global random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

class BaseDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        self.X, self.y = preprocess_data.turn_X_y_2_tensor(preprocess_data.process_data_X_y(root_data_path=data_path))

    def __len__(self):
        return len(self.y)

class StandardDataset(BaseDataset):
    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        return self.X[index], self.y[index]

class AttentionDataset(BaseDataset):
    def __init__(self, data_path: str, hidden_size: int = 64) -> None:
        super().__init__(data_path)
        self.hidden_size = hidden_size
        self.embedding = self.create_embedding(self.X.shape[1])
        self.X = self.process_data_for_attention(self.X)

    def create_embedding(self, num_features: int) -> torch.nn.Embedding:
        torch.manual_seed(RANDOM_SEED)  # Set seed for embedding initialization
        return torch.nn.Embedding(num_features, self.hidden_size)

    def process_data_for_attention(self, data: torch.Tensor) -> torch.Tensor:
        batch_size, num_features = data.shape
        feature_indices = torch.arange(num_features).unsqueeze(0).repeat(batch_size, 1)
        embedded_features = self.embedding(feature_indices)
        return embedded_features * data.unsqueeze(-1)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]

class preprocess_data:
    def __init__(self, 
                 root_data_path: str,
                 train_path: str = None,
                 valid_path: str = None,
                 test_path: str = None,
                 batch_size: int = None,
                 dataset_type: str = 'standard',
                 hidden_size: int = 64,
                 random_seed: int = 42) -> None:
        self.root_data_path = root_data_path
        self.train_path = train_path if train_path else root_data_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.dataset_type = dataset_type
        self.hidden_size = hidden_size
        self.random_seed = random_seed

    @staticmethod
    def process_data_X_y(root_data_path:str) -> Generator:
        all_label_data = pd.DataFrame()
        for single_label_data_path in os.listdir(root_data_path):
            all_label_data = pd.concat(
                [all_label_data, pd.read_csv(os.path.join(root_data_path, single_label_data_path))], axis=0
            )

        def make_num_label(df, label):
            array = df[label].unique()
            index_array = np.arange(array.shape[0])
            return df[label].apply(lambda x : index_array[array == x][0] if x in array else index_array.shape[0])

        all_label_data = all_label_data.drop(['loaded_uid', 'host_age', 'BMI'], axis = 1).reset_index(drop=True)
        all_label_data['sex'] = all_label_data['sex'].apply(lambda x : 1 if x == 'Female' else 0 if x == 'Male' else 2)

        all_label_data['country'] = make_num_label(all_label_data, 'country')
        all_label_data['label'] = make_num_label(all_label_data, 'label')

        numeric_columns = all_label_data.select_dtypes(include=[np.number]).columns
        all_label_data[numeric_columns] = all_label_data[numeric_columns].replace(0, 1e-12)

        return (all_label_data[column] for column in all_label_data.columns)
    
    @staticmethod
    def turn_X_y_2_tensor(X_y:Generator):
        first_flag = True
        for column_data in X_y:
            if first_flag:
                xy = torch.tensor(column_data, dtype=torch.float32).view(-1,1)
                first_flag = False
            else:
                xy = torch.cat([xy, torch.tensor(column_data, dtype=torch.float32).view(-1,1)], dim = -1)
        return xy[:, :-1], xy[:, -1].to(torch.int32)

    @staticmethod
    def create_weighted_sampler(labels: torch.tensor, target_ratio: float = 2.0) -> WeightedRandomSampler:
        class_counts = torch.bincount(labels)
        target_counts = class_counts.clone()
        
        # 设置目标样本数，将少数类的样本数增加到多数类的 1/target_ratio
        max_count = class_counts.max().item()
        for i in range(len(target_counts)):
            if class_counts[i] < max_count / target_ratio:
                target_counts[i] = int(max_count / target_ratio)
        
        weights = target_counts.float() / class_counts.float()
        sample_weights = weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=int(target_counts.sum().item()),
            replacement=True
        )
        return sampler

    @staticmethod
    def get_label_counts(dataset: Dataset) -> dict:
        """
        Count the number of samples for each label in the dataset.
        
        Args:
            dataset (Dataset): The dataset to analyze.
        
        Returns:
            dict: A dictionary with labels as keys and their counts as values.
        """
        label_counts = {}
        for _, label in dataset:
            label = label.item()
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
        return label_counts

    def get_dataset(self, data_path: str) -> Dataset:
        if self.dataset_type == 'standard':
            return StandardDataset(data_path)
        elif self.dataset_type == 'attention':
            return AttentionDataset(data_path, self.hidden_size)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def get_dataloader(self) -> Tuple[DataLoader]:
        if not self.valid_path:
            all_dataset = self.get_dataset(self.train_path)
            if not self.test_path:
                train_dataset, valid_dataset, test_dataset = random_split(
                    all_dataset, 
                    [0.7, 0.15, 0.15],
                    generator=torch.Generator().manual_seed(RANDOM_SEED)
                )
                train_labels = torch.tensor([all_dataset[i][1] for i in train_dataset.indices])
            else:
                train_dataset, valid_dataset = random_split(
                    all_dataset, 
                    [0.85, 0.15],
                    generator=torch.Generator().manual_seed(RANDOM_SEED)
                )
                test_dataset = self.get_dataset(self.test_path)
                train_labels = torch.tensor([all_dataset[i][1] for i in train_dataset.indices])
        elif self.test_path:
            train_dataset = self.get_dataset(self.train_path)
            valid_dataset = self.get_dataset(self.valid_path)
            test_dataset = self.get_dataset(self.test_path)
            train_labels = torch.tensor([label for _, label in train_dataset])
        else:
            train_dataset = self.get_dataset(self.train_path)
            valid_dataset, test_dataset = random_split(
                self.get_dataset(self.valid_path), 
                [0.5, 0.5],
                generator=torch.Generator().manual_seed(RANDOM_SEED)
            )
            train_labels = torch.tensor([label for _, label in train_dataset])
        
        train_sampler = preprocess_data.create_weighted_sampler(train_labels)
        valid_sampler = preprocess_data.create_weighted_sampler(torch.tensor([y for _, y in valid_dataset]))
        test_sampler = preprocess_data.create_weighted_sampler(torch.tensor([y for _, y in test_dataset]))

        if not self.batch_size:
            train_batch_size = len(train_sampler)
            valid_batch_size = len(valid_sampler)
            test_batch_size = len(test_sampler)
        else:
            train_batch_size = valid_batch_size = test_batch_size = self.batch_size
        
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, sampler=train_sampler)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, sampler=valid_sampler)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, sampler=test_sampler)

        return train_dataloader, valid_dataloader, test_dataloader

def get_attention_dataloader(data_path: str, batch_size: int, hidden_size: int = 64) -> DataLoader:
    dataset = AttentionDataset(data_path, hidden_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(RANDOM_SEED))

if __name__ == '__main__':
    data_path = 'data/standardized'
    batch_size = 32
    hidden_size = 64

    # Using standard dataset
    standard_preprocessor = preprocess_data(root_data_path=data_path, batch_size=batch_size, dataset_type='standard')
    train_dataloader, valid_dataloader, test_dataloader = standard_preprocessor.get_dataloader()

    # Using attention dataset
    attention_preprocessor = preprocess_data(root_data_path=data_path, batch_size=batch_size, dataset_type='attention', hidden_size=hidden_size)
    train_dataloader_attn, valid_dataloader_attn, test_dataloader_attn = attention_preprocessor.get_dataloader()

    def print_dataloader_distribution(dataloader: DataLoader, name: str):
        label_counts = {}
        for _, labels in dataloader:
            for label in labels:
                label = label.item()
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
        print(f"{name} dataset distribution after balancing:")
        print(label_counts)
        print(f"Total samples: {sum(label_counts.values())}")
        print()

    print_dataloader_distribution(train_dataloader, "Train")
    print_dataloader_distribution(valid_dataloader, "Validation")
    print_dataloader_distribution(test_dataloader, "Test")

    print_dataloader_distribution(train_dataloader_attn, "Train (Attention)")
    print_dataloader_distribution(valid_dataloader_attn, "Validation (Attention)")
    print_dataloader_distribution(test_dataloader_attn, "Test (Attention)")
