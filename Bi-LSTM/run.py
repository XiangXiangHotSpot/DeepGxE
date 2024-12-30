
import os
import sys
import time
import torch
import uuid

import random
import datetime
import argparse
import tempfile

import numpy as np
import torch.optim as optim

from tqdm import tqdm
from models import EarlyFusion
from scipy.stats import pearsonr
from torch.utils.tensorboard import SummaryWriter
from dataset.dataloader import training_scenario_split, get_dataset
from utils import parameter_printing, results_printing, prediction_record, get_model_checkpoint_path, model_save

def get_hyper_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SEED", type=int, default=1)                               # Random seed
    parser.add_argument("--k_fold_SEED", type=int, default=42)                       # Random seed for k-fold
    parser.add_argument("--fold", type=int, default=5)                               # Number of folds
    parser.add_argument("--model", type=str, default="Bi-LSTM")                      # Model Name
    parser.add_argument("--epoch", type=int, default=150)                            # Number of epochs
    parser.add_argument("--batch_size", type=int, default=256)                       # Mini-batch size
    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument("--nesterov", default=True, action="store_true")             # Use Nesterov momentum
    parser.add_argument("--learning_rate", type=float, default=0.005)                # Initial learning rate
    parser.add_argument("--scheduler_factor", type=float, default=0.5)               # Learning rate adjustment factor
    parser.add_argument("--weight_decay", type=float, default=0.001)                 # Weight decay
    parser.add_argument("--patience", type=int, default=5)                           # Patience for learning rate adjustment
    parser.add_argument("--dropout", type=float, default=0.3)                        # Dropout rate
    parser.add_argument("--slope", type=float, default=0.1)                          # LeakyReLU slope
    parser.add_argument("--momentum", type=float, default=0.9)                       # Momentum
    parser.add_argument("--vocab_size_snp", type=int, default=3)                     # SNP vocabulary size, set to 3 for only 0, 1 and 2
    parser.add_argument("--embedding_dim_snp", type=int, default=16)                 # Embedding dimension
    parser.add_argument("--channel", type=int, nargs='+', default=[32, 32, 32, 32])  # Number of channels
    parser.add_argument("--kernel_size", type=int, nargs='+', default=[9, 9, 9, 9])  # Convolution kernel size
    parser.add_argument("--stride", type=int, nargs='+', default=[2, 2, 2, 2])       # Stride size
    parser.add_argument("--training_scenario", type=str, default="S1")               # Training scenario: S1, S2, S3
    parser.add_argument("--pheno_normalize", type=int, default=0)         #  Whether to normalize phenotype data
    parser.add_argument("--phenotype", type=str, choices=["Yield", "Ph"], default="Yield") # Phenotype

    # Bi-LSTM configuration
    parser.add_argument("--input_size", type=int, default=9)                         # Bi-LSTM input size, 9 weather variabels
    parser.add_argument("--hidden_size", type=int, default=64)                       # Hidden layers
    parser.add_argument("--num_layers", type=int, default=2)                         # Number of layers
    parser.add_argument("--output_size", type=int, default=16)                       # Output Dimension
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()
    args.start_time = datetime.datetime.now().strftime('%m-%d %H:%M:%S')

    return args


def mse_computing(array1, array2):
    return np.mean(np.square(np.asarray(array1) - np.asarray(array2)))


def get_model(config, current_fold, checkpoint_path):
    model = EarlyFusion(config).to(config.device)

    if current_fold < 2 :
        torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))

    return model


def get_optimizer(config, model):
    if config.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
            )
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=config.learning_rate, momentum=config.momentum,
            weight_decay=config.weight_decay, nesterov=config.nesterov
            )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=config.scheduler_factor, verbose=1, min_lr=0, patience=config.patience
    )

    return optimizer, scheduler


def k_fold_cross_validation(config):
    all_max_cor = []
    all_max_cor_epoch = []
    all_last_cor = []
    all_min_MSE = []
    all_min_MSE_epoch = []
    all_last_MSE = []

    checkpoint_path = get_model_checkpoint_path()

    dataset = get_dataset(config)

    for fold in range(config.fold):
        current_fold = fold + 1

        train_loader, val_loader, max_pheno_value, min_pheno_value, mean_pheno_value, std_pheno_value = training_scenario_split(dataset, fold, config)

        model = get_model(config, current_fold, checkpoint_path)
        optimizer, scheduler = get_optimizer(config, model)

        tb_writer = SummaryWriter(
            log_dir='tensorboard_log' + '/' + config.phenotype + '_' +
            config.training_scenario + '_fold_' + str(current_fold) +  '_' +
            time.strftime('%m-%d_%H.%M', time.localtime()) + '_' + str(uuid.uuid4())
            )

        max_cor, max_cor_epoch, min_MSE, min_MSE_epoch, cor, MSE = train(
            config, model, optimizer, scheduler, train_loader, val_loader, 
            current_fold, max_pheno_value, min_pheno_value,
            mean_pheno_value, std_pheno_value, tb_writer
            )
        
        all_max_cor.append(max_cor)
        all_max_cor_epoch.append(max_cor_epoch)
        all_last_cor.append(cor) 
        all_min_MSE.append(min_MSE)
        all_min_MSE_epoch.append(min_MSE_epoch)
        all_last_MSE.append(MSE)

        model_save(config, model, current_fold, cor)
 
    if os.path.exists(checkpoint_path) is True:
        os.remove(checkpoint_path)

    tb_writer.close()

    return all_max_cor, all_max_cor_epoch, all_last_cor, all_min_MSE, all_min_MSE_epoch, all_last_MSE


def evaluation(
        model, val_loader, max_cor, max_cor_epoch, min_mse,
        min_mse_epoch, max_pheno_value, min_pheno_value,
        mean_pheno_value, std_pheno_value, config
        ):
    model.eval()

    predict_array = []
    target_array = []

    for test_iter, val_batch_data in enumerate(val_loader):
        genotype = val_batch_data['genotype']
        environment = val_batch_data['weather']
        seq_len = val_batch_data['seq_len']

        if config.pheno_normalize == 1:
            phenotype = val_batch_data['normalized_phenotype']
            phenotype = phenotype * std_pheno_value + mean_pheno_value
        else:
            phenotype = val_batch_data['phenotype']

        genotype = genotype.to(config.device)
        environment = environment.to(config.device)

        genotype = genotype.unsqueeze(dim=1)

        with torch.no_grad():
          predict = model(genotype, environment, seq_len, config).squeeze(dim=1)

        if config.pheno_normalize == 1:
            predict = predict * std_pheno_value + mean_pheno_value

        predict_array.extend(predict.detach().cpu().numpy())
        target_array.extend(phenotype.detach().numpy())

    cor, _ = pearsonr(predict_array, target_array)
    MSE = mse_computing(predict_array, target_array)

    print("[cor :", cor, end='] ')
    print("[MSE :", MSE, end='] ')
    print("[max cor :", max_cor, end='] ')
    print("[max cor epoch :", max_cor_epoch, end='] ')
    print("[min MSE :", min_mse, end='] ')
    print("[min MSE epoch :", min_mse_epoch, end='] ')
    print("\n")

    return cor, MSE, predict_array, target_array


def train(
        config, model, optimizer, scheduler, train_loader, val_loader,
        current_fold, max_pheno_value, min_pheno_value, 
        mean_pheno_value, std_pheno_value, tb_writer
        ): 
    max_cor = -1.0
    max_cor_epoch = 0
    min_MSE = 10000
    min_MSE_epoch = 0

    all_predictions_list = []

    loss_function = torch.nn.MSELoss(reduction='none')

    for epoch_index in tqdm(range(config.epoch)):
        print(
            "A Bi-LSTM Model in Scenario %s ----Fold %s -- Epoch %s Is Training---------" % (
                config.training_scenario, current_fold, epoch_index + 1
                )
            )

        epoch_loss = 0
        epoch_size = 0

        for train_iter, train_batch_data in enumerate(train_loader):
            model.train()
            genotype = train_batch_data['genotype']
            environment = train_batch_data['weather']
            seq_len = train_batch_data['seq_len']

            if config.pheno_normalize == 1:
                phenotype = train_batch_data['normalized_phenotype']
            else:
                phenotype = train_batch_data['phenotype']

            epoch_size = epoch_size + genotype.shape[0]

            genotype = genotype.to(config.device)
            environment = environment.to(config.device)
            phenotype = phenotype.to(config.device)
 
            genotype = genotype.unsqueeze(dim=1)
            phenotype = phenotype.unsqueeze(dim=1)

            output = model(genotype, environment, seq_len, config)

            loss = loss_function(input=output, target=phenotype).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / epoch_size
        print("loss:", epoch_loss)

        cor, MSE, predict_array, target_array = evaluation(
            model, val_loader, max_cor, max_cor_epoch,min_MSE,
            min_MSE_epoch, max_pheno_value, min_pheno_value,
            mean_pheno_value, std_pheno_value, config
            )

        scheduler.step(epoch_loss)

        tags = ["Train loss", "Correlation", "MSE", "Learning rate"]
        tb_writer.add_scalar(tags[0], epoch_loss, epoch_index+1)
        tb_writer.add_scalar(tags[1], cor,  epoch_index+1)
        tb_writer.add_scalar(tags[2], MSE,  epoch_index+1)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"],  epoch_index+1)

        all_predictions_list.append(predict_array)

        if epoch_index + 1 == config.epoch:
            all_predictions_list.append(target_array)
            prediction_record(all_predictions_list, current_fold, config, cor)

        if cor > max_cor:
            max_cor = cor
            max_cor_epoch = epoch_index+1

        if MSE < min_MSE:
            min_MSE = MSE
            min_MSE_epoch = epoch_index+1

    return max_cor, max_cor_epoch, min_MSE, min_MSE_epoch, cor, MSE


if __name__ == '__main__':
    torch.set_printoptions(profile="full")
    np.set_printoptions(threshold=np.inf)

    config = get_hyper_parameter()
    parameter_printing(config)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    all_max_cor, all_max_cor_epoch, \
    all_last_cor, all_min_MSE, \
    all_min_MSE_epoch, all_last_MSE = k_fold_cross_validation(config)

    results_printing(
        config, all_max_cor, all_max_cor_epoch,
        all_last_cor, all_min_MSE,
        all_min_MSE_epoch, all_last_MSE
        )