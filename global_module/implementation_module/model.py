import tensorflow as tf

from global_module.settings_module import set_params, set_dir


class DeepAttentionClassifier:
    def __init__(self, params, dir_obj):
        self.params = params
        self.dir_obj = dir_obj
        self.call_pipeline()

    def call_pipeline(self):
        self.create_network_pipeline()

    def create_network_pipeline(self):
        self.create_placeholders()
        self.create_rnn_cell()
        self.embedding_layer_lookup()
        self.run_rnn()
        if self.params.use_attention:
            self.apply_attention()
        self.compute_cost()

        if self.params.mode == 'TR':
            self.train()

    def create_placeholders(self):
        with tf.variable_scope('embedding_variable'):
            self.word_emb_matrix = tf.get_variable("word_embedding_matrix", shape=[self.params.vocab_size, self.params.EMB_DIM], dtype=tf.float32)
        with tf.variable_scope('placeholders'):
            self.word_input = tf.placeholder(name="word_input", shape=[self.params.batch_size, self.params.MAX_SEQ_LEN], dtype=tf.int32)
            self.seq_length = tf.placeholder(name="seq_len", shape=[self.params.batch_size], dtype=tf.int32)
            self.label = tf.placeholder(name="labels", shape=[self.params.batch_size], dtype=tf.int32)

    def create_rnn_cell(self):
        if self.params.rnn_cell == 'lstm':
            with tf.variable_scope('lstm_cell'):
                self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.params.RNN_HIDDEN_DIM, forget_bias=1.0)
                self.rnn_cell = tf.contrib.rnn.DropoutWrapper(self.rnn_cell, input_keep_prob=self.params.keep_prob, output_keep_prob=self.params.keep_prob)
        else:
            with tf.variable_scope('gru_cell'):
                self.rnn_cell = tf.contrib.rnn.GRUCell(num_units=self.params.RNN_HIDDEN_DIM)
                self.rnn_cell = tf.contrib.rnn.DropoutWrapper(self.rnn_cell, input_keep_prob=self.params.keep_prob, output_keep_prob=self.params.keep_prob)

    def embedding_layer_lookup(self):
        with tf.variable_scope('lookup'):
            self.word_emb_feature = tf.nn.embedding_lookup(self.word_emb_matrix,
                                                           self.word_input,
                                                           name='word_emb_feature',
                                                           validate_indices=True)

    def run_rnn(self):
        with tf.variable_scope('rnn_block'):
            if not self.params.bidirectional:
                self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(cell=self.rnn_cell,
                                                                     inputs=self.word_emb_feature,
                                                                     sequence_length=self.seq_length,
                                                                     dtype=tf.float32)
            else:
                ((fw_outputs, bw_outputs), (fw_state, bw_state)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.rnn_cell,
                                                                                                   cell_bw=self.rnn_cell,
                                                                                                   inputs=self.word_emb_feature,
                                                                                                   sequence_length=self.seq_length,
                                                                                                   dtype=tf.float32)
                self.rnn_outputs = tf.concat(values=(fw_outputs, bw_outputs), axis=2, name='concat_output')

                if self.params.rnn_cell == 'lstm':
                    self.rnn_state = tf.concat(values=(fw_state[1], bw_state[1]), axis=1, name='concat_state')
                    # self.rnn_state = fw_state[1] + bw_state[1]
                elif self.params.rnn_cell == 'gru':
                    self.rnn_state = tf.concat(values=(fw_state, bw_state), axis=1, name='concat_state')
                    # self.rnn_state = fw_state + bw_state

    def apply_attention(self):
        with tf.variable_scope('attention'):
            attention_vector = tf.get_variable(name='attention_vector',
                                               shape=[self.params.ATTENTION_DIM],
                                               dtype=tf.float32)

            mlp_layer_projection = tf.layers.dense(inputs=self.rnn_outputs,
                                                   units=self.params.ATTENTION_DIM,
                                                   activation=tf.nn.tanh,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   name='mlp_projection')

            attended_vector = tf.tensordot(mlp_layer_projection, attention_vector, axes=[[2], [0]])
            attention_weights = tf.expand_dims(tf.nn.softmax(attended_vector), -1)

            weighted_input = tf.matmul(self.rnn_outputs, attention_weights, transpose_a=True)
            self.attention_output = tf.squeeze(weighted_input, axis=2)

    def compute_cost(self):

        with tf.variable_scope('dense_layers'):
            with tf.variable_scope('dropout'):
                if self.params.use_attention:
                    sentence_vector = tf.nn.dropout(self.attention_output, keep_prob=self.params.keep_prob, name='attention_vector_dropout')
                else:
                    sentence_vector = tf.nn.dropout(self.rnn_state, keep_prob=self.params.keep_prob, name='rnn_state_dropout')

            output1 = tf.layers.dense(inputs=sentence_vector,
                                      units=self.params.num_classes,
                                      kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                                      bias_initializer=tf.constant_initializer(0.01),
                                      name='layer_1')

        with tf.variable_scope('last_layer'):
            logits = tf.layers.dense(inputs=output1,
                                     units=self.params.num_classes,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.constant_initializer(0.01),
                                     name='logit_layer')

            self.prediction = tf.cast(tf.argmax(input=logits, axis=1, name='prediction'), dtype=tf.int32)
            self.probabilities = tf.nn.softmax(logits, name='softmax_probability')

            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    correct_prediction = tf.equal(tf.cast(tf.argmax(self.prediction, 0), tf.int32), self.label)
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.variable_scope('compute_loss'):
            with tf.variable_scope('softmax_loss'):
                gold_labels = tf.one_hot(indices=self.label, depth=self.params.num_classes, name='gold_label')
                softmax_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=gold_labels, logits=logits), name='reg_loss')

            with tf.variable_scope('reg_loss'):
                if self.params.mode == 'TR':
                    tvars = tf.trainable_variables()
                    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.params.REG_CONSTANT, scope=None)
                    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, tvars)
                    reg_penalty_word_emb = tf.contrib.layers.apply_regularization(l2_regularizer, [self.word_emb_matrix])
                    reg_loss = regularization_penalty - reg_penalty_word_emb
                else:
                    reg_loss = 0

            self.loss = softmax_loss + reg_loss

            if self.params.mode == 'TR' and self.params.log:
                self.train_loss = tf.summary.scalar('loss_train', self.loss)
                self.train_accuracy = tf.summary.scalar('acc_train', self.accuracy)
            elif self.params.mode == 'VA' and self.params.log:
                valid_loss = tf.summary.scalar('loss_valid', self.loss)
                valid_accuracy = tf.summary.scalar('acc_valid', self.accuracy)
                self.merged_else = tf.summary.merge([valid_loss, valid_accuracy])
            else:
                self.merged_else = []

            print 'Loss Computation: DONE'

    def train(self):
        with tf.variable_scope('train'):
            self._lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()

            self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.params.max_grad_norm, name='global_norm')
            optimizer = tf.train.GradientDescentOptimizer(self.lr, name='optimizer')
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-2, name='optimizer')
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._lr, epsilon=1e-6, name='optimizer')
            grads_and_vars = optimizer.compute_gradients(self.loss)
            self._train_op = optimizer.apply_gradients(zip(self.grads, tvars), name='apply_gradient')

            if self.params.log:
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                self.merged_train = tf.summary.merge([self.train_loss, self.train_accuracy, grad_summaries_merged])
            else:
                self.merged_train = []

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
    classifier_obj = DeepAttentionClassifier(params, dir_obj)


if __name__ == '__main__':
    main()
