import tensorflow as tf
import numpy as np
import random
import sys
import time

from global_module.settings_module import set_params, set_dir, set_dict
from global_module.implementation_module import model
from global_module.implementation_module import reader


# merged = tf.summary.merge_all()

def run_epoch(session, eval_op, min_cost, model_obj, dict_obj, epoch_num, verbose=False):
    combined_loss = -1.0
    epoch_combined_loss = 0.0
    total_correct = 0.0
    total_instances = 0.0
    print('\nrun epoch')

    params = model_obj.params
    dir_obj = model_obj.dir_obj
    data_filename = dir_obj.data_filename
    label_filename = dir_obj.label_filename

    for step, (input_seq_arr, label_arr) \
            in enumerate(reader.data_iterator(params, data_filename, label_filename, model_obj.params.indices, dict_obj)):
        feed_dict = {}
        feed_dict[model_obj.word_input] = input_seq_arr
        feed_dict[model_obj.label] = label_arr

        loss, prediction, probabilities, _ = session.run([model_obj.loss,
                                                          model_obj.prediction,
                                                          model_obj.probabilities,
                                                          # model_obj.curr_accuracy,
                                                          eval_op],
                                                         feed_dict=feed_dict)

        total_correct += np.sum(prediction == label_arr)
        total_instances += params.batch_size
        epoch_combined_loss += loss

    print 'Epoch Num: %d, CE loss: %.4f, Accuracy: %.4f' % (epoch_num, (epoch_combined_loss), (total_correct / total_instances) * 100)

    if (params.mode == 'VA'):
        model_saver = tf.train.Saver()
        print('**** Current minimum on valid set: %.4f ****' % (min_cost))

        if (epoch_combined_loss < min_cost):
            min_cost = epoch_combined_loss
            model_saver.save(session, save_path=dir_obj.model_path + dir_obj.model_name, latest_filename=dir_obj.latest_checkpoint)
            print('==== Model saved! ====')

    return epoch_combined_loss, min_cost, min_cost


def getLength(fileName):
    print('Reading :', fileName)
    dataFile = open(fileName, 'r')
    count = 0
    for _ in dataFile:
        count += 1
    dataFile.close()
    return count, np.arange(count)


def run_train(dict_obj):
    mode_train, mode_valid, mode_all = 'TR', 'VA', 'ALL'

    # train object

    params_train = set_params.ParamsClass(mode=mode_train)
    dir_train = set_dir.Directory(mode_train)
    params_train.num_instances, params_train.indices = getLength(dir_train.data_filename)

    # valid object

    params_valid = set_params.ParamsClass(mode=mode_valid)
    dir_valid = set_dir.Directory(mode_valid)
    params_valid.num_instances, params_valid.indices = getLength(dir_valid.data_filename)

    params_train.num_classes = params_valid.num_classes = len(dict_obj.label_dict)

    if (params_train.enable_shuffle):
        random.shuffle(params_train.indices)
        random.shuffle(params_valid.indices)

    min_loss = sys.float_info.max

    word_emb_path = dir_train.word_embedding
    word_emb_matrix = np.float32(np.genfromtxt(word_emb_path, delimiter=' '))
    params_train.vocab_size = params_valid.vocab_size = len(word_emb_matrix)

    print('***** INITIALIZING TF GRAPH *****')

    with tf.Graph().as_default(), tf.Session() as session:
        # train_writer = tf.summary.FileWriter(dir_train.log_path + '/train', session.graph)
        # test_writer = tf.summary.FileWriter(dir_train.log_path + '/test')

        # random_normal_initializer = tf.random_normal_initializer()
        # random_uniform_initializer = tf.random_uniform_initializer(-params_train.init_scale, params_train.init_scale)
        xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

        with tf.name_scope('train'):
            with tf.variable_scope("model", reuse=None, initializer=xavier_initializer):
                train_obj = model.CNNClassification(params_train, dir_train)

        with tf.name_scope('valid'):
            with tf.variable_scope("model", reuse=True, initializer=xavier_initializer):
                valid_obj = model.CNNClassification(params_valid, dir_valid)

        if not params_train.enable_checkpoint:
            session.run(tf.global_variables_initializer())

        if params_train.enable_checkpoint:
            ckpt = tf.train.get_checkpoint_state(dir_train.model_path)
            if ckpt and ckpt.model_checkpoint_path:
                print("Loading model from: %s" % ckpt.model_checkpoint_path)
                tf.train.Saver().restore(session, ckpt.model_checkpoint_path)
        elif (not params_train.use_random_initializer):
            session.run(tf.assign(train_obj.word_emb_matrix, word_emb_matrix, name="word_embedding_matrix"))

        print('**** TF GRAPH INITIALIZED ****')

        # train_writer.add_graph(tf.get_default_graph())

        start_time = time.time()
        for i in range(params_train.max_max_epoch):
            lr_decay = params_train.lr_decay ** max(i - params_train.max_epoch, 0.0)
            train_obj.assign_lr(session, params_train.learning_rate * lr_decay)

            # print(params_train.learning_rate * lr_decay)

            print('\n++++++++=========+++++++\n')

            print("Epoch: %d Learning rate: %.5f" % (i + 1, session.run(train_obj.lr)))
            train_loss, _, summary = run_epoch(session, train_obj.train_op, min_loss, train_obj, dict_obj, i, verbose=True)
            print("Epoch: %d Train loss: %.3f" % (i + 1, train_loss))

            valid_loss, curr_loss, summary = run_epoch(session, tf.no_op(), min_loss, valid_obj, dict_obj, i)
            if (curr_loss < min_loss):
                min_loss = curr_loss

            print("Epoch: %d Valid loss: %.3f" % (i + 1, valid_loss))

            curr_time = time.time()
            print('1 epoch run takes ' + str(((curr_time - start_time) / (i + 1)) / 60) + ' minutes.')

            # train_writer.close()
            # test_writer.close()


def main():
    dict_obj = set_dict.Dictionary()
    run_train(dict_obj)


if __name__ == "__main__":
    main()
