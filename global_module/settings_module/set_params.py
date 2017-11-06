class ParamsClass:
    def __init__(self, mode='TR'):
        """
        :param mode: 'TR' for train, 'TE' for test, 'VA' for valid
        """
        self.mode = mode
        self.init_scale = 0.1
        self.learning_rate = 0.01
        self.max_grad_norm = 10
        self.max_epoch = 100
        self.max_max_epoch = 150
        self.rnn_cell = 'lstm'  # or 'gru' OR 'lstm'
        self.bidirectional = True

        if mode == 'TR':
            self.keep_prob = 0.5
        else:
            self.keep_prob = 1.0

        self.lr_decay = 0.99

        self.enable_shuffle = False
        self.enable_checkpoint = False
        self.all_lowercase = True
        self.log = False
        self.log_step = 9

        if mode == 'TE':
            self.enable_shuffle = False

        self.NUM_LAYER = 10
        self.REG_CONSTANT = 0.001
        self.MAX_SEQ_LEN = 20
        self.EMB_DIM = 300
        self.RNN_HIDDEN_DIM = 50
        self.ATTENTION_DIM = 512

        self.batch_size = 64
        self.vocab_size = 30
        self.is_word_trainable = True

        self.use_unknown_word = True
        self.use_random_initializer = False

        self.use_attention = False

        self.indices = None
        self.num_instances = None
        self.num_classes = None
        self.sampling_threshold = 1
