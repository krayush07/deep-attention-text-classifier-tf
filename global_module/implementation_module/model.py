import tensorflow as tf

from global_module.settings_module import set_params, set_dir


class CNNClassification():
    def __init__(self, params, dir_obj):
        self.params = params
        self.dir_obj = dir_obj
        self.call_pipeline()

    def call_pipeline(self):
        self.create_placeholders()
        self.create_network_pipeline()
        self.compute_cost()
        if self.params.mode == 'TR':
            self.train()

    def create_placeholders(self):
        self.word_emb_matrix = tf.get_variable("word_embedding_matrix", shape=[self.params.vocab_size, self.params.EMB_DIM], dtype=tf.float32)
        self.word_input = tf.placeholder(name="word_input", shape=[self.params.batch_size, self.params.MAX_SEQ_LEN], dtype=tf.int32)
        self.label = tf.placeholder(name="labels", shape=[self.params.batch_size], dtype=tf.int32)

    def convolution_layer(self, input, filter, stride, padding, activation, name):
        with tf.variable_scope(name):
            weights = tf.get_variable(name='weights',
                                      shape=filter,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
                                      # initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))

            biases = tf.get_variable(name='biases',
                                     shape=filter[-1],
                                     initializer=tf.constant_initializer(0.1))

            # TODO: (done) --> check the data alignment (NHWC)
            conv = tf.nn.conv2d(name="convolution",
                                input=input,
                                filter=weights,
                                strides=stride,
                                padding=padding)

            if (activation == 'RELU'):
                return tf.nn.relu(tf.nn.bias_add(conv, biases), name="relu")
            elif (activation == 'TANH'):
                return tf.nn.tanh(tf.nn.bias_add(conv, biases), name="tanh")

    def max_pool(self, input, pool_size, stride, padding, name):
        with tf.variable_scope(name):
            return tf.nn.max_pool(name="avg_pool",
                                  value=input,
                                  ksize=pool_size,
                                  strides=stride,
                                  padding=padding)

    def avg_pool(self, input, pool_size, stride, padding, name):
        with tf.variable_scope(name):
            return tf.nn.avg_pool(name="avg_pool",
                                  value=input,
                                  ksize=pool_size,
                                  strides=stride,
                                  padding=padding)

    def pooling_layer(self, input, pool_size, stride, padding, pool_option, name):
        if (pool_option == 'MAX'):
            return self.max_pool(input, pool_size, stride, padding, name)
        elif (pool_option == 'AVG'):
            return self.avg_pool(input, pool_size, stride, padding, name)

    def conv_pool_block(self, layer_num, input):
        pooled_output = []
        num_filters = self.params.num_filters[layer_num]
        for i in range(len(self.params.filter_width[layer_num])):
            filter_width = self.params.filter_width[layer_num][i]

            conv_output = self.convolution_layer(input=input,
                                                 filter=[filter_width, self.params.EMB_DIM, 1, num_filters],
                                                 stride=[1, 1, 1, 1],
                                                 padding=self.params.conv_padding,
                                                 activation=self.params.conv_activation,
                                                 name="conv_layer_" + str(layer_num) + '_' + str(filter_width))

            pool_output = self.pooling_layer(input=conv_output,
                                             pool_size=[1, self.params.pool_width[layer_num], 1, 1],
                                             # self.params.pool_stride[layer_num]
                                             stride=[1, self.params.MAX_SEQ_LEN + 5 - filter_width + 1, 1, 1],
                                             padding=self.params.pool_padding,
                                             pool_option=self.params.pool_option,
                                             name="pool_layer_" + str(layer_num))

            pooled_output.append(pool_output)

        return tf.concat(pooled_output, axis=1)
        # return tf.reshape(tf.concat(pooled_output, axis=1), [self.params.batch_size, -1])

    def create_network_pipeline(self):
        padded_input = tf.pad(self.word_input, paddings=[[0, 0], [5, 0]])        #constant paading of N at start
        padded_input = self.word_input
        self.input_matrix = tf.nn.embedding_lookup(self.word_emb_matrix, padded_input, name="emb_lookup")
        self.sent_input_matrix = tf.expand_dims(self.input_matrix, -1)

        curr_input = self.sent_input_matrix
        curr_output = curr_input
        for i in range(len(self.params.num_filters)):
            curr_output = self.conv_pool_block(i, curr_input)

        pooled_output_flat = tf.reshape(curr_output, shape=[self.params.batch_size, -1], name='pooled_falt')
        pooled_output_flat = tf.nn.dropout(pooled_output_flat, keep_prob=self.params.keep_prob)

        # dense_layer1 = tf.layers.dense(inputs=pooled_output_flat, units=1024, activation=tf.nn.relu, name='dense_layer_1')
        # self.last_layer = tf.layers.dense(inputs=dense_layer1, units=self.params.num_classes, activation=tf.nn.relu, name='dense_layer_2')
        #
        # # self.last_layer = dense_layer1
        # self.last_layer = tf.nn.dropout(self.last_layer, keep_prob=self.params.keep_prob)
        self.last_layer = pooled_output_flat

    def compute_cost(self):
        logits = tf.layers.dense(inputs=self.last_layer,
                                 units=self.params.num_classes,
                                 kernel_initializer=tf.orthogonal_initializer,
                                 bias_initializer=tf.constant_initializer(0.1),
                                 name='logits')
        # gold_labels = tf.one_hot(indices=self.label, depth=self.params.num_classes)

        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=logits))

        if (self.params.mode == 'TR'):
            tvars = tf.trainable_variables()
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.params.REG_CONSTANT, scope=None)
            regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, tvars)
            reg_penalty_word_emb = tf.contrib.layers.apply_regularization(l2_regularizer, [self.word_emb_matrix])
            self.loss = self.loss + regularization_penalty - reg_penalty_word_emb

        self.prediction = tf.cast(tf.argmax(input=logits, axis=1, name='prediction'), dtype=tf.int32)
        self.probabilities = tf.nn.softmax(logits, name='softmax_probability')

        # self.curr_accuracy = tf.contrib.metrics.accuracy(self.prediction, self.label, name='accuracy')

        print 'Loss Computation: DONE'

    def train(self):
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.params.max_grad_norm, name='global_norm')
        # optimizer = tf.train.GradientDescentOptimizer(self.lr, name='optimizer')
        # optimizer = tf.train.AdamOptimizer(learning_rate=1e-2, name='optimizer')
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._lr, epsilon=1e-6, name='optimizer')
        self._train_op = optimizer.apply_gradients(zip(self.grads, tvars), name='apply_gradient')

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


def main():
    params = set_params.ParamsClass(mode='TR')
    dir_obj = set_dir.Directory('TR')
    multi_view_obj = CNNClassification(params, dir_obj)


if __name__ == '__main__':
    main()
