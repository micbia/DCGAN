import configparser, numpy as np

def StringOrNone(string):
    try:
        return eval(string)
    except:
        return string

class DefaultConfig:
    def __init__(self, PATH):
        self.path = PATH

        def_config = configparser.ConfigParser()
        def_config.optionxform=str
        def_config['TRAINING'] = {'INPUT_DIM'       : 100,
                                  'COARSE_DIM'      : [7, 7, 256],
                                  'OUTPUT_DIM'      : [28, 28, 1],
                                  'KERNEL_SIZE'     : 5,
                                  'UPSAMPLE_SIZE'   : 2,
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
        self.input_dim      = eval(trainconfig['INPUT_DIM'])
        self.coarse_dim     = np.array(eval(trainconfig['COARSE_DIM']), dtype=int)
        self.output_dim     = np.array(eval(trainconfig['OUTPUT_DIM']), dtype=int)
        self.upsampl_size   = eval(trainconfig['UPSAMPLE_SIZE'])
        self.kernel_size    = eval(trainconfig['KERNEL_SIZE'])
        self.batch_size     = eval(trainconfig['BATCH_SIZE'])
        self.alpha          = eval(trainconfig['ALPHA'])
        self.dropout        = eval(trainconfig['DROPOUT'])
        self.lr             = eval(trainconfig['LEARNING_RATE'])
        self.beta1          = eval(trainconfig['BETA1'])
