import configparser, numpy as np

def StringOrNone(string):
    ''' convert initial condition that are 'None' in proper python none'''
    try:
        return eval(string)
    except:
        return string

class DefaultConfig:
    def __init__(self, PATH):
        self.path = PATH

        def_config = configparser.ConfigParser()
        def_config.optionxform=str
        def_config['TRAINING'] = {'DATASET'         : 'mnist',
                                  'INPUT_DIM'       : 100,
                                  'COARSE_DIM'      : [7, 7, 256],
                                  'OUTPUT_DIM'      : [28, 28, 1],
                                  'KERNEL_SIZE'     : 5,
                                  'UPSAMPLE_SIZE'   : 2,
                                  'EPOCHS'          : 10,
                                  'BATCH_SIZE'      : 32,
                                  'ALPHA'           : 0.01,
                                  'DROPOUT'         : 0.5,
                                  'LEARNING_RATE'   : 2e-4,
                                  'BETA1'           : 0.5}

        def_config['RESUME'] = {'RESUME_PATH'      : None,
                                'RESUME_EPOCH'     : 0}

        with open(self.path+'/example.ini', 'w') as configfile:
            def_config.write(configfile)


class NetworkConfig:
    def __init__(self, CONFIG_FILE):
        self.config_file    = CONFIG_FILE

        config = configparser.ConfigParser()
        config.read(self.config_file)
        
        trainconfig = config['TRAINING']
        self.type_of_gan    = trainconfig['TYPE_GAN']
        self.dataset        = trainconfig['DATASET']
        self.input_dim      = eval(trainconfig['INPUT_DIM'])
        self.coarse_dim     = np.array(eval(trainconfig['COARSE_DIM']), dtype=int)
        self.output_dim     = np.array(eval(trainconfig['OUTPUT_DIM']), dtype=int)
        self.stride         = eval(trainconfig['STRIDE'])
        self.filters        = eval(trainconfig['FILTERS'])
        self.epochs         = eval(trainconfig['EPOCHS'])
        self.batch_size     = eval(trainconfig['BATCH_SIZE'])
        self.dropout        = eval(trainconfig['DROPOUT'])
        
        resumeconfig = config['RESUME']
        self.resume_path    = StringOrNone(resumeconfig['RESUME_PATH'])
        self.resume_epoch   = eval(resumeconfig['RESUME_EPOCH'])
