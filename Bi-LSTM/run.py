
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
    parser.add_argument("--SEED", type=int, default=1)                              # 随机种子
    parser.add_argument("--k_fold_SEED", type=int, default=1)                              # 随机种子
    parser.add_argument("--fold", type=int, default=5)                              # 折数
    parser.add_argument("--model", type=str, default="EarlyFusion", choices=['EarlyFusion', 'LayerWise'])
    parser.add_argument("--epoch", type=int, default=150)                           # epoch数
    parser.add_argument("--batch_size", type=int, default=128)                      # mini-batch大小
    parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument("--nesterov", default=True, action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.005)               # 初始学习率
    parser.add_argument("--scheduler_factor", type=float, default=0.5)              # 学习率调整力度
    parser.add_argument("--weight_decay", type=float, default=0.001)                # 权重衰减
    parser.add_argument("--patience", type=int, default=5)                          # 若超过相应epoch数量loss没降将调整相应学习率
    parser.add_argument("--dropout", type=float, default=0.3)                       # 随机失活 1
    parser.add_argument("--slope", type=float, default=0.1)                         # leakyReLu斜率
    parser.add_argument("--momentum", type=float, default=0.9)                      # 动量
    parser.add_argument("--vocab_size_snp", type=int, default=3)                    # SNP 词表大小，只有0和1 设为: 2
    parser.add_argument("--embedding_dim_snp", type=int, default=16)                # SNP 词向量维度
    parser.add_argument("--channel", type=int, nargs='+', default=[32, 32, 32, 32]) # 通道数
    parser.add_argument("--kernel_size", type=int, nargs='+', default=[9, 9, 9, 9]) # 卷积核大小
    parser.add_argument("--stride", type=int, nargs='+', default=[2, 2, 2, 2])      # 步长大小
    parser.add_argument("--training_scenario", type=str, default="S1")              # S1 S2 S3
    parser.add_argument("--pheno_normalize", type=int, default=0)                   # 是否对表型数据做归一化
    parser.add_argument("--weather_normalize", type=int, default=1)                 # 是否对天气数据做归一化

    parser.add_argument("--phenotype", type=str, choices=["Yield", "Ph"], default="Yield")              #表型
    parser.add_argument("--weather_type", type=str, choices=["dayily", "weekly"], default="weekly")              #天气

    parser.add_argument("--input_size", type=int, default=9)                        # 输入维度大小，这里指的是共九个变量
    parser.add_argument("--hidden_size", type=int, default=64)                      # 隐藏层
    parser.add_argument("--num_layers", type=int, default=2)                        # 层数
    parser.add_argument("--output_size", type=int, default=16)                      # 输出维度
    
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
        optimizer, 'max', factor=config.scheduler_factor, verbose=1, patience=config.patience
    )

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min', factor=config.scheduler_factor, verbose=1, min_lr=0, patience=config.patience
    # )

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

    # Split data into K folds and create data loaders for each fold
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
        genotype = val_batch_data['genotype'] # torch.Size([16, 18239])
        environment = val_batch_data['weather'] # torch.Size([16, 9])
        seq_len = val_batch_data['seq_len'] # [16] cpu

        if config.pheno_normalize == 0:
            phenotype = val_batch_data['phenotype'] # torch.Size([16])

        if config.pheno_normalize == 1:
            phenotype = val_batch_data['normalized_phenotype'] # torch.Size([16])
            phenotype = phenotype * std_pheno_value + mean_pheno_value

        genotype = genotype.to(config.device)
        environment = environment.to(config.device)

        # SNP 数据和环境型数据增加一个维度
        genotype = genotype.unsqueeze(dim=1)

        with torch.no_grad():
          predict = model(genotype, environment, seq_len, config).squeeze(dim=1)

        if config.pheno_normalize == 1:
            # predict = predict * (max_pheno_value - min_pheno_value) + min_pheno_value
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

    # training start
    for epoch_index in tqdm(range(config.epoch)):  # 循环
        print(
            "A Bi-LSTM Model in Scenario %s ----Fold %s -- Epoch %s Is Training---------" % (
                config.training_scenario, current_fold, epoch_index + 1
                )
            )

        epoch_loss = 0
        epoch_size = 0

        for train_iter, train_batch_data in enumerate(train_loader):
            model.train()
            genotype = train_batch_data['genotype'] # torch.Size([16, 18239])
            environment = train_batch_data['weather'] # torch.Size([16, 9])
            seq_len = train_batch_data['seq_len']

            if config.pheno_normalize == 0:
                phenotype = train_batch_data['phenotype'] # torch.Size([16])
            elif config.pheno_normalize == 1:
                phenotype = train_batch_data['normalized_phenotype'] # torch.Size([16])

            epoch_size = epoch_size + genotype.shape[0]

            genotype = genotype.to(config.device)
            environment = environment.to(config.device)
            phenotype = phenotype.to(config.device)
 
            # SNP 数据和环境型数据增加一个维度
            genotype = genotype.unsqueeze(dim=1)
            phenotype = phenotype.unsqueeze(dim=1)

            # 输入到网络获得预测值 output
            output = model(genotype, environment, seq_len, config)

            # 预测值和标签算损失
            loss = loss_function(input=output, target=phenotype).sum()

            # 网络参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加每个 iter 损失
            epoch_loss += loss.item()

        # 计算一个 Epoch 的 loss
        epoch_loss = epoch_loss / epoch_size
        print("loss:", epoch_loss)

        cor, MSE, predict_array, target_array = evaluation(
            model, val_loader, max_cor, max_cor_epoch,min_MSE,
            min_MSE_epoch, max_pheno_value, min_pheno_value,
            mean_pheno_value, std_pheno_value, config
            )

        # 根据 loss 判断是否需要更新 LR
        scheduler.step(cor)

        tags = ["Train loss", "Correlation", "MSE", "Learning rate"]
        tb_writer.add_scalar(tags[0], epoch_loss, epoch_index+1)
        tb_writer.add_scalar(tags[1], cor,  epoch_index+1)
        tb_writer.add_scalar(tags[2], MSE,  epoch_index+1)
        tb_writer.add_scalar(tags[3], optimizer.param_groups[0]["lr"],  epoch_index+1)

        #记录每个epoch的预测结果和
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

    # 获得超参数
    config = get_hyper_parameter()
    parameter_printing(config)

    # 确定种子
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Five-fold Cross Validation

    all_max_cor, all_max_cor_epoch, \
    all_last_cor, all_min_MSE, \
    all_min_MSE_epoch, all_last_MSE = k_fold_cross_validation(config)

    results_printing(
        config, all_max_cor, all_max_cor_epoch,
        all_last_cor, all_min_MSE,
        all_min_MSE_epoch, all_last_MSE
        )