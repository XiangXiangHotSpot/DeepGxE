import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data.dataset import Subset


class MultiModalDataset(Dataset):
    def __init__(self, geno_path, weather_path, config):
        self.data = pd.read_csv(geno_path)

        self.phenotype = np.array(self.data['Value'].values)
        
        mean_value = np.mean(self.phenotype)
        std_value = np.std(self.phenotype)
        self.normalized_phenotype = torch.FloatTensor(
            (self.phenotype - mean_value) / std_value
            )

        self.phenotype = torch.FloatTensor(self.phenotype)
        
        start_col = self.data.columns.get_loc('S1A_1145442')
        
        self.genotype = torch.LongTensor(self.data.iloc[:, start_col:].values)

        self.weather = np.load(weather_path, allow_pickle=True)  
        self.seq_len = [np.count_nonzero(np.any(matrix != 0, axis=1)) for matrix in self.weather]
        self.weather = torch.FloatTensor(self.weather)

        self.gid = self.data["Gid"]

        self.occ_loc_cycle = (
            self.data["Occ"].astype(str) + '_' +
            self.data['Loc_no'].astype(str) + '_' +
            self.data['Cycle'].astype(str)
                                 )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        genetic_sample = self.genotype[idx]
        weather_sample = self.weather[idx]
        seq_len_sample = self.seq_len[idx]
        label_sample = self.phenotype[idx]
        normalized_label_sample = self.normalized_phenotype[idx]
        gid_sample = self.gid[idx]
        occ_loc_cycle_sample = self.occ_loc_cycle[idx]

        return {'genotype' : genetic_sample,
                'weather' : weather_sample,
                'phenotype' : label_sample,
                'seq_len' : seq_len_sample,
                'normalized_phenotype' : normalized_label_sample,
                'gid' : gid_sample,
                'occ_loc_cycle' : occ_loc_cycle_sample}
    
    def get_max_min_phenotype(self):
        max_value = self.phenotype.max()
        min_value = self.phenotype.min()

        return max_value.item(), min_value.item()

    def get_mean_std_phenotype(self):
        mean_value = self.phenotype.mean()
        std_value = self.phenotype.std()

        return mean_value.item(), std_value.item()


def get_dataset(config):
    if config.phenotype == "Yield" :
        data_path = '../WheatYieldProcessing/2_Geno/output/YieldGeno.csv' 
        weather_path = '../WheatYieldProcessing/3_Env/output/YieldWeeklyWeatherNormalized.pkl' 

    elif config.phenotype == "Ph" :
        data_path = '../PlantHeightProcessing/2_Geno/output/PhGeno.csv'
        weather_path = '../PlantHeightProcessing/3_Env/output/PhWeeklyWeatherNormalized.pkl'

    dataset = MultiModalDataset(data_path, weather_path, config)

    return dataset


def perform_kfold_split(data, n_splits, seed):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kfold.split(data))


def training_scenario_split(dataset, current_fold, config):
    max_pheno_value, min_pheno_value = dataset.get_max_min_phenotype()
    mean_pheno_value, std_pheno_value = dataset.get_mean_std_phenotype()

    if config.phenotype == "Yield" :
        unique_gid_path = '../WheatYieldProcessing/1_Pheno/output/UniqueGid.csv'
        unique_occ_loc_cycle_path = '../WheatYieldProcessing/1_Pheno/output/UniqueOccLocCycle.csv'
    elif config.phenotype == "Ph" :
        unique_gid_path = '../PlantHeightProcessing/1_Pheno/output/UniqueGid.csv'
        unique_occ_loc_cycle_path = '../PlantHeightProcessing/1_Pheno/output/UniqueOccLocCycle.csv'

    if config.training_scenario == "S1":
        splits = perform_kfold_split(dataset, config.fold, config.k_fold_SEED)
        train_data_indices, test_data_indices = splits[current_fold]

    elif config.training_scenario == "S2":
        unique_gid = pd.read_csv(unique_gid_path)
        gid = dataset.gid
        
        splits = perform_kfold_split(unique_gid, config.fold, config.k_fold_SEED)
        train_idx, test_idx = splits[current_fold]

        train_gids = unique_gid['Gid'].iloc[train_idx]
        test_gids = unique_gid['Gid'].iloc[test_idx]

        train_data_indices = gid[gid.isin(train_gids)].index
        test_data_indices = gid[gid.isin(test_gids)].index

    elif config.training_scenario == "S3":
        unique_occ_loc_cycle = pd.read_csv(unique_occ_loc_cycle_path)
        occ_loc_cycle = dataset.occ_loc_cycle

        splits = perform_kfold_split(unique_occ_loc_cycle, config.fold, config.k_fold_SEED)
        train_idx, test_idx = splits[current_fold]

        train_occ_loc_cycle = unique_occ_loc_cycle['Occ_Loc_no_Cycle'].iloc[train_idx]
        test_occ_loc_cycle = unique_occ_loc_cycle['Occ_Loc_no_Cycle'].iloc[test_idx]

        train_data_indices = occ_loc_cycle[occ_loc_cycle.isin(train_occ_loc_cycle)].index
        test_data_indices = occ_loc_cycle[occ_loc_cycle.isin(test_occ_loc_cycle)].index

    else:
        raise ValueError("Current Training Scenario Does not Exist, Please Recheck the Inputs ")
    
    train_dataset = Subset(dataset, train_data_indices)
    val_dataset = Subset(dataset, test_data_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor = 2
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor = 2
        )

    return train_loader, val_loader, max_pheno_value, min_pheno_value, mean_pheno_value, std_pheno_value