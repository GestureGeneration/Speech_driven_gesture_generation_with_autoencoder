from configargparse import ArgumentParser, YAMLConfigFileParser
import os
# Modify this function to set your home directory for this repo
def home_out(path):
    return os.path.join(os.environ['HOME'], 'tmp', 'MoCap', path)

def construct_config_parser():
    parser = ArgumentParser(args_for_setting_config_path = ['-config'],
                            default_config_files = ['./config.yaml'],
                            config_file_parser_class = YAMLConfigFileParser)

    parser.add('--seed', help='Random seed')

    # ---- The data directories ----

    parser.add('-data_dir', '--data_dir', required=True,
               help='The directory with the preprocessed dataset')
    parser.add('--summary_dir', default=home_out('summaries_exp'),
               help='Directory for saving the summary data')
    parser.add('--chkpt_dir', default=home_out('chkpts_exp'),
               help='Directory for saving the model checkpoints')
    parser.add('--results_file', default=home_out('results.txt'),
               help='File for saving the results of the experiments')

    # ---- Flags ----

    parser.add('-pretrain', '--pretrain_network', action='store_true', 
               help='If set, pretrain the model in a layerwise manner')
    parser.add('-load_model', '--load_model', action='store_true',
               help='If set, load the model from a checkpoint')
    parser.add('-no_early_stopping', '--no_early_stopping', action='store_false',
               help='If set, disable early stopping')

    # ---- Network architecture --- 

    parser.add('--frame_size', help='Dimensionality of the input for a single frame')
    parser.add('--num_hidden_layers', help='The number of hidden layers')
    parser.add('--middle_layer', help='The size of the middle layer')

    parser.add('--layer1_width', help='The number of hidden layers')
    parser.add('--layer2_width', help='The number of hidden layers')
    parser.add('--layer3_width', help='The number of hidden layers')

    # ---- Training parameters ----

    parser.add('--chunk_length', help='Length of the chunks during data processing')
    parser.add('--noise_variance', help='Variance of the Gaussian noise that is added' + \
                                        'to every motion frame during training')
    parser.add('--training_epochs', help='Number of training epochs')
    parser.add('--pretraining_epochs', help='Number of pretraining epochs')

    parser.add('--batch_size', help='Batch size')
    parser.add('--lr', help='Learning rate for training')
    parser.add('--pretraining_lr', help='Learning rate for pretraining')
    parser.add('--weight_decay', help='The multiplier for weight decay, which can be' + \
                                      'turned off by setting this parameter to None')
    parser.add('--dropout_keep_prob', help='Probability of keeping a weight')

    parser.add('--delta_for_early_stopping', 
               help='Controls the sensitivity of early stopping.' + \
                    'A delta of 0.05 means that the training is stopped when' + \
                    'the loss is 5 percent worse than the best loss so far.')
    
    return parser