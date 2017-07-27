class ParamsClass():
    def __init__(self, mode):
        """
        :param mode: 'TR' for train, 'TE' for test, 'VA' for valid
        """
        self.mode = mode
        self.init_scale = 0.1
        self.learning_rate = 0.001
        self.max_grad_norm = 5
        self.max_epoch = 2
        self.max_max_epoch = 5

        if (mode == 'TR'):
            self.keep_prob = 0.6
        else:
            self.keep_prob = 1.0

        self.lr_decay = 0.95

        self.enable_shuffle = False
        self.enable_checkpoint = False

        if (mode == 'TE'):
            self.enable_shuffle = False

        self.REG_CONSTANT = 0.005
        self.MAX_SEQ_LEN = 50
        self.EMB_DIM = 300

        self.batch_size = 32
        self.vocab_size = 30
        self.is_word_trainable = True

        self.use_unknown_word = True
        self.use_random_initializer = False

        self.indices = None
        self.num_instances = None
        self.num_classes = None

        ''' PARAMS FOR CONV BLOCK '''
        self.num_filters = [256]
        self.filter_width = [[2, 3, 5]]
        self.conv_activation = 'RELU'
        self.conv_padding = 'VALID'

        self.pool_width = [5]
        self.pool_stride = [2]
        self.pool_padding = 'VALID'
        self.pool_option = 'MAX'
