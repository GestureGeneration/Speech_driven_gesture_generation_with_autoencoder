from configargparse import ArgumentParser, YAMLConfigFileParser
import os

# NOTE: the global variable 'args' for accessing the config parameters from other modules
#       is defined at the very bottom of this file
def construct_config_parser():
    parser = ArgumentParser(args_for_setting_config_path = ['-config'],
                            default_config_files = ['./config.yaml'],
                            config_file_parser_class = YAMLConfigFileParser)

    parser.add('--seed', type=int, help='Random seed')

    # ---- Data directories ----
    
    parser.add('--data_dir',     help='The directory with the preprocessed dataset')
    parser.add('--summary_dir',  help='Directory for saving the summary data')
    parser.add('--chkpt_dir',    help='Directory for saving the model checkpoints')
    parser.add('--results_file', help='File for saving the results of the experiments')

    # ---- Input/output files for 'decode.py' only ----

    parser.add('-input_file',  default=None, 
               help="The encoded prediction file that will be decoded (only used in 'decode.py')")
    parser.add('-output_file', default=None,
               help="The output file where the decoded gesture will be stored (only used in 'decode.py')")

    # ---- Flags ----

    parser.add('-pretrain', '--pretrain_network',               action='store_true', 
               help='If set, pretrain the model in a layerwise manner')
    parser.add('-load_model', '--load_model_from_checkpoint',   action='store_true',
               help='If set, load the model from a checkpoint')
    parser.add('-no_early_stopping', '--no_early_stopping',     action='store_false',
               help='If set, disable early stopping')

    # ---- Network architecture --- 

    parser.add('--frame_size',         type=int,   help='Dimensionality of the input for a single frame')
    parser.add('--num_hidden_layers',  type=int,   help='The number of hidden layers')
    parser.add('--middle_layer',       type=int,   help='The size of the middle layer')
   
    parser.add('--layer1_width',       type=int,   help='The number of hidden layers')
    parser.add('--layer2_width',       type=int,   help='The number of hidden layers')
    parser.add('--layer3_width',       type=int,   help='The number of hidden layers')

    # ---- Training parameters ----

    parser.add('--chunk_length',       type=int,   help='Length of the chunks during data processing')
    parser.add('--dropout_keep_prob',  type=float, help='Probability of keeping a weight')
    parser.add('--weight_decay',       type=float, help='The multiplier for weight decay, which can be' + \
                                                        'turned off by setting this parameter to None')
    parser.add('--noise_variance',     type=float, help='Variance of the Gaussian noise that is added' + \
                                                        'to every motion frame during training')
    
    parser.add('--training_epochs',    type=int,   help='Number of training epochs')
    parser.add('--pretraining_epochs', type=int,   help='Number of pretraining epochs')
    parser.add('--batch_size',         type=int,   help='Batch size')
    parser.add('--lr',                 type=float, help='Learning rate for training')
    parser.add('--pretraining_lr',     type=float, help='Learning rate for pretraining')
    
    parser.add('--delta_for_early_stopping', type=float,
               help='Controls the sensitivity of early stopping.' + \
                    'A delta of 0.05 means that the training is stopped when' + \
                    'the loss is 5 percent worse than the best loss so far.')
    
    return parser

# Any file that imports this module will have access to this global variable.
args = construct_config_parser().parse_args()