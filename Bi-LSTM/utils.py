import os
import csv
import uuid
import time
import torch
import tempfile
import datetime


def parameter_printing(args):
    args.start_time = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
    print("Start Time           :", args.start_time )
    print("SEED                 = {}".format(args.SEED))
    print("Fold                 = {}".format(args.fold))
    print("Model                = {}".format(args.model))
    print("Total epoch          = {}".format(args.epoch))
    print("Batch size           = {}".format(args.batch_size))
    print("Optimizer            = {}".format(args.optimizer))
    print("Learning rate        = {}".format(args.learning_rate))
    print("Scheduler factor     = {}".format(args.scheduler_factor))
    print("Weight decay         = {}".format(args.weight_decay))
    print("Patience             = {}".format(args.patience))
    print("Dropout              = {}".format(args.dropout))
    print("Momentum             = {}".format(args.momentum))
    print("LeakyRelu slope      = {}".format(args.slope))
    print("Vocab size SNP       = {}".format(args.vocab_size_snp))
    print("Embedding dim SNP    = {}".format(args.embedding_dim_snp))
    print("Channel              = {}".format(args.channel))
    print("Kernel size          = {}".format(args.kernel_size))
    print("Stride               = {}".format(args.stride))
    print("Training Scenario    = {}".format(args.training_scenario))

    print("Pheno Normalize      = {}".format(args.pheno_normalize))
    print("Phenotype            = {}".format(args.phenotype))

    print("Input Size           = {}".format(args.input_size))
    print("Hidden Size          = {}".format(args.hidden_size))
    print("Num Layers           = {}".format(args.num_layers))
    print("Output Size          = {}".format(args.output_size))


def results_printing(config, max_cor, max_cor_epoch, last_cor, min_MSE, min_MSE_epoch, last_MSE):
    file = open('result/results.txt', 'a')
    file.write('\n')
    file.write('\n')
    file.write('+++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    file.write('Start Time            = {} \n'.format(config.start_time))
    file.write('End Time              = {} \n'.format(datetime.datetime.now().strftime('%m-%d %H:%M:%S')))

    file.write('SEED                  = {} \n'.format(config.SEED))
    file.write('Model                 = {} \n'.format(config.model))
    file.write('Fold                  = {} \n'.format(config.fold))
    file.write('Total epoch           = {} \n'.format(config.epoch))
    file.write('Batch size            = {} \n'.format(config.batch_size))
    file.write('Optimizer             = {} \n'.format(config.optimizer))
    file.write('Learning rate         = {} \n'.format(config.learning_rate))
    file.write('Scheduler factor      = {} \n'.format(config.scheduler_factor))
    file.write('Weight decay          = {} \n'.format(config.weight_decay))
    file.write('Patience              = {} \n'.format(config.patience))
    file.write('Dropout               = {} \n'.format(config.dropout))
    file.write('LeakyRelu slope       = {} \n'.format(config.slope))
    file.write('Momentum              = {} \n'.format(config.momentum))
    file.write('Vocab size SNP        = {} \n'.format(config.vocab_size_snp))
    file.write('Embedding dim SNP     = {} \n'.format(config.embedding_dim_snp))
    file.write('Channel               = {} \n'.format(config.channel))
    file.write('Kernel size           = {} \n'.format(config.kernel_size))
    file.write('Stride                = {} \n'.format(config.stride))
    file.write('Training Scenario     = {} \n'.format(config.training_scenario))

    file.write('Pheno Normalize       = {} \n'.format(config.pheno_normalize))
    file.write('Phenotype             = {} \n'.format(config.phenotype))

    file.write('Input Size            = {} \n'.format(config.input_size))
    file.write('Hidden Size           = {} \n'.format(config.hidden_size))
    file.write('Num Layers            = {} \n'.format(config.num_layers))
    file.write('Output Size           = {} \n'.format(config.output_size))
        
    file.write('Max Correlation        = {} \n'.format(max_cor))
    file.write('Max Correlation Epoch  = {} \n'.format(max_cor_epoch))
    file.write('MIN MSE:               = {} \n'.format(min_MSE))
    file.write('MIN MSE Epoch          = {} \n'.format(min_MSE_epoch))
    file.write('Last Correlation       = {} \n'.format(last_cor))
    file.write('Mean MAX Correlation   = {} \n'.format(sum(max_cor) / len(max_cor)))
    file.write('Mean Last Correlation  = {} \n'.format(sum(last_cor) / len(last_cor)))
    file.write('Last MSE               = {} \n'.format(last_MSE))
    file.write('Mean Last MSE          = {} \n'.format(sum(last_MSE) / len(last_MSE)))

    file.write('\n')

    file.write('+++++++++++++++++++\n')
    file.close()


def prediction_record(epoch_predictions_list, current_fold, config, cor):
    csv_folder_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    full_csv_filename =  "../PredictedArray/" + csv_folder_name 
    
    if not os.path.exists(full_csv_filename):
        os.mkdir(full_csv_filename)

    csv_filename = (
        full_csv_filename + "/" +
        config.phenotype + "_" +
        config.training_scenario + "_" +
        "fold" + str(current_fold) + "_" +
        str(cor) + ".csv" 
        )
    
    column_names = ["epoch"]

    column_names = [f"epoch_{i+1}" for i in range(config.epoch)] + ["target_array"]
    transposed_epoch_predictions = list(map(list, zip(*epoch_predictions_list)))

    with open(csv_filename, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(column_names)
        
        for row in transposed_epoch_predictions:
            csv_writer.writerow(row)
    return


def model_save(config, model, current_fold, cor):
    folder_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    full_folder_name = "../ModelPth/" + folder_name
    
    if not os.path.exists(full_folder_name):
        os.mkdir(full_folder_name)

    pth_file_name = (
        full_folder_name + "/" +
        config.phenotype + "_" +
        config.training_scenario + "_" +
        "fold" + str(current_fold) + "_" +
        str(round(cor, 5)) + "_" +
        str(uuid.uuid4()) + ".pth"
        )

    torch.save(model.state_dict(), pth_file_name)

    return None


def get_model_checkpoint_path():
    unique_id = str(uuid.uuid4())
    timestamp = time.strftime('%m-%d_%H.%M', time.localtime())
    checkpoint_filename = f"{timestamp}_{unique_id}_Bi-LSTM.pt"
    checkpoint_path = os.path.join(tempfile.gettempdir(), checkpoint_filename)

    return checkpoint_path