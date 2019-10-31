import configparser

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
        def_config['TRAINING'] = {'INPUT_DIM' :     4,
                                  'OUTPUT_DIM' :    1,
                                  'LOSS' :          'mse',
                                  'HIDDEN_DIM_1' :  10,
                                  'HIDDEN_DIM_2' :  '',
                                  'HIDDEN_DIM_3' :  '',
                                  'BATCH_SIZE' :    32,
                                  'EPOCHS' :        1000,
                                  'DROPOUT' :       0.2,
                                  'ACTIVATION' :    'elu',
                                  'LEARNING_RATE' : 0.01,
                                  'METRICS' :       'mae, mse',
                                  'REGULARIZER'  :  None,                                  
                                  'GPU' :           None,
                                  'PATH' :          None,
                                  'DATA_USED' :     None}

        def_config['RESUME'] = { 'RESUME_PATH'  :   None,
                                 'RESUME_EPOCH' :   0}

        with open(self.path+'/example.ini', 'w') as configfile:
            def_config.write(configfile)


class NetworkConfig:
    def __init__(self, CONFIG_FILE):
        self.config_file    = CONFIG_FILE

        config = configparser.ConfigParser()
        config.read(self.config_file)
        
        trainconfig = config['TRAINING']
        self.input_dim      = eval(trainconfig['INPUT_DIM'])
        self.output_dim     = eval(trainconfig['OUTPUT_DIM'])
        self.hidden_dim_1   = eval(trainconfig['HIDDEN_DIM_1'])
        self.hidden_dim_2   = eval(trainconfig['HIDDEN_DIM_2'])
        self.hidden_dim_3   = eval(trainconfig['HIDDEN_DIM_3'])

        self.loss           = trainconfig['LOSS']
        self.batch_size     = eval(trainconfig['BATCH_SIZE'])
        self.epochs         = eval(trainconfig['EPOCHS'])
        self.dropout        = eval(trainconfig['DROPOUT'])
        self.activation     = trainconfig['ACTIVATION']
        self.lr             = eval(trainconfig['LEARNING_RATE'])
        self.metrics        = trainconfig['METRICS'].split(', ')
        self.data_used      = trainconfig['DATA_USED']
        self.regularizers   = trainconfig['REGULARIZER']
        self.gpu            = eval(trainconfig['GPU'])
        
        resumeconfig = config['RESUME']
        self.resume_epoch   = eval(resumeconfig['RESUME_EPOCH'])
        self.resume_path    = StringOrNone(resumeconfig['RESUME_PATH'])
        
