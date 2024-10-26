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
    def create_weighted_sampler(labels: torch.tensor) -> WeightedRandomSampler:
        class_counts = torch.bincount(labels)
        weights = 1. / class_counts.float()
        sample_weights = weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True
        )
        return sampler

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
                test_dataset = bac_dataset(test_dataset)
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
        if not self.batch_size:
            train_batch_size = len(train_dataset)
            valid_batch_size = len(valid_dataset)
            test_batch_size = len(test_dataset)
        else:
            train_batch_size = valid_batch_size = test_batch_size = self.batch_size
        
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, sampler=train_sampler)
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size)

        return train_dataloader, valid_dataloader, test_dataloader
    

if __name__ == '__main__':
    print('test with root_data_path and train path')
    train_dataloader, valid_dataloder, test_dataloader = preprocess_data(root_data_path='data/standardized', train_path='data/standardized').get_dataloader()
    for x,y in train_dataloader:
        print(x.shape)
        print(y.shape)
    print('test without train_path only with root_data_path')
    train_dataloader, valid_dataloder, test_dataloader = preprocess_data(root_data_path='data/standardized').get_dataloader()
    for x,y in train_dataloader:
        print(x.shape)
        print(y.shape)