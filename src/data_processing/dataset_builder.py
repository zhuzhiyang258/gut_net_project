import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from typing import Tuple, Generator

class bac_dataset(Dataset):
    def __init__(self, data_path:str) -> None:
        self.X, self.y = preprocess_data.turn_X_y_2_tensor(preprocess_data.process_data_X_y(root_data_path=data_path))
    
    def __getitem__(self, index) -> Tuple[torch.tensor]:
        X, y = self.X[index], self.y[index]
        return X, y
    
    def __len__(self):
        return len(self.y)

class preprocess_data:
    def __init__(self, 
                root_data_path:str, 
                train_path:str = None, 
                valid_path:str = None, 
                test_path:str = None, 
                batch_size:int = None, 
                random_seed:int = 42) -> None:
        '''
            Return three Dataloader: train dataloader, valid dataloader, test dataloader. And the X data shape is with 2 dims, y data shape is with 1 dim for each dataloader.
            Args: 
                root_data_path : the path you must set.
                train_path : if you do not set it, it will be the same as root_data_path.
                valid_path, test_path : feel free to set or not.
                batch_size : the batch size(samples num) when you start a iteration in dataloader.
                random_seed : fix a random seed to make it repeatable.
        '''
        self.root_data_path = root_data_path
        if train_path == None:
            self.train_path = root_data_path
        else:
            self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.batch_size = batch_size
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
            if label == 'label':
                print("\nClass to label mapping:")
                for i, class_name in enumerate(array):
                    print(f"Class '{class_name}' -> Label {i}")
            return df[label].apply(lambda x : index_array[array == x][0] if x in array else index_array.shape[0])

        all_label_data = all_label_data.drop(['loaded_uid', 'host_age', 'BMI'], axis = 1).reset_index(drop=True)
        all_label_data['sex'] = all_label_data['sex'].apply(lambda x : 1 if x == 'Female' else 0 if x == 'Male' else 2)

        all_label_data['country'] = make_num_label(all_label_data, 'country')
        all_label_data['label'] = make_num_label(all_label_data, 'label')

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
    def get_label_counts(dataset: bac_dataset) -> dict:
        """
        Count the number of samples for each label in the dataset.
        
        Args:
            dataset (bac_dataset): The dataset to analyze.
        
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

    def get_dataloader(self) -> Tuple[DataLoader]:
        if not self.valid_path:
            all_dataset = bac_dataset(self.train_path)
            if not self.test_path:
                train_dataset, valid_dataset, test_dataset = random_split(
                all_dataset, 
                [0.7, 0.15, 0.15],
                generator=torch.Generator().manual_seed(self.random_seed)
            )
                train_labels = torch.tensor([all_dataset[i][1] for i in train_dataset.indices])
            else:
                train_dataset, valid_dataset = random_split(
                all_dataset, 
                [0.85, 0.15],
                generator=torch.Generator().manual_seed(self.random_seed)
            )
                test_dataset = bac_dataset(self.test_path)
                train_labels = torch.tensor([all_dataset[i][1] for i in train_dataset.indices])


        elif self.test_path:
            train_dataset = bac_dataset(self.train_path)
            valid_dataset = bac_dataset(self.valid_path)
            test_dataset = bac_dataset(self.test_path)
            train_labels = []
            for _, label in train_dataset:
                train_labels.append(label)
            train_labels = torch.tensor(train_labels)
        
        else:
            train_dataset = bac_dataset(self.train_path)
            valid_dataset, test_dataset = random_split(
                bac_dataset(self.valid_path), 
                [0.5, 0.5],
                generator=torch.Generator().manual_seed(self.random_seed)
            )
            train_labels = []
            for _, label in train_dataset:
                train_labels.append(label)
            train_labels = torch.tensor(train_labels)
        
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

        # Check for overlap between datasets
        train_indices = set(train_dataset.indices if hasattr(train_dataset, 'indices') else range(len(train_dataset)))
        valid_indices = set(valid_dataset.indices if hasattr(valid_dataset, 'indices') else range(len(valid_dataset)))
        test_indices = set(test_dataset.indices if hasattr(test_dataset, 'indices') else range(len(test_dataset)))

        train_valid_overlap = train_indices.intersection(valid_indices)
        train_test_overlap = train_indices.intersection(test_indices)
        valid_test_overlap = valid_indices.intersection(test_indices)

        if train_valid_overlap:
            print(f"Overlap between train and valid datasets: {train_valid_overlap}")
        else:
            print("No overlap between train and valid datasets.")

        if train_test_overlap:
            print(f"Overlap between train and test datasets: {train_test_overlap}")
        else:
            print("No overlap between train and test datasets.")

        if valid_test_overlap:
            print(f"Overlap between valid and test datasets: {valid_test_overlap}")
        else:
            print("No overlap between valid and test datasets.")

        self.print_label_distribution(train_dataset, valid_dataset, test_dataset)

        return train_dataloader, valid_dataloader, test_dataloader

    def print_label_distribution(self, train_dataset, valid_dataset, test_dataset):
        def get_label_counts(dataset):
            labels = torch.tensor([y for _, y in dataset])
            unique_labels, counts = torch.unique(labels, return_counts=True)
            return dict(zip(unique_labels.tolist(), counts.tolist()))

        train_counts = get_label_counts(train_dataset)
        valid_counts = get_label_counts(valid_dataset)
        test_counts = get_label_counts(test_dataset)

        print("\nLabel distribution:")
        print(f"{'Label':<10}{'Train':<10}{'Valid':<10}{'Test':<10}")
        print("-" * 40)
        
        all_labels = sorted(set(train_counts.keys()) | set(valid_counts.keys()) | set(test_counts.keys()))
        
        for label in all_labels:
            train_count = train_counts.get(label, 0)
            valid_count = valid_counts.get(label, 0)
            test_count = test_counts.get(label, 0)
            print(f"{label:<10}{train_count:<10}{valid_count:<10}{test_count:<10}")

        print("\nTotal samples:")
        print(f"{'Train':<10}{sum(train_counts.values()):<10}")
        print(f"{'Valid':<10}{sum(valid_counts.values()):<10}")
        print(f"{'Test':<10}{sum(test_counts.values()):<10}")

if __name__ == '__main__':
    print('test with root_data_path and train path')
    train_dataloader, valid_dataloader, test_dataloader = preprocess_data(root_data_path='data/standardized', train_path='data/standardized').get_dataloader()
    for x,y in train_dataloader:
        print(x.shape)
        print(y.shape)
    print('test without train_path only with root_data_path')
    train_dataloader, valid_dataloder, test_dataloader = preprocess_data(root_data_path='data/standardized').get_dataloader()
    for x,y in train_dataloader:
        print(x.shape)
        print(y.shape)
        
    print("\nTrain dataset label counts after balancing:")
    train_label_counts = {}
    for _, y in train_dataloader:
        for label in y:
            label = label.item()
            if label not in train_label_counts:
                train_label_counts[label] = 0
            train_label_counts[label] += 1
    print(train_label_counts)
    
    print("\nValidation dataset label counts after balancing:")
    valid_label_counts = {}
    for _, y in valid_dataloader:
        for label in y:
            label = label.item()
            if label not in valid_label_counts:
                valid_label_counts[label] = 0
            valid_label_counts[label] += 1
    print(valid_label_counts)
    
    print("\nTest dataset label counts after balancing:")
    test_label_counts = {}
    for _, y in test_dataloader:
        for label in y:
            label = label.item()
            if label not in test_label_counts:
                test_label_counts[label] = 0
            test_label_counts[label] += 1
    print(test_label_counts)
