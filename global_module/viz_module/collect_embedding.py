import tensorflow as tf
from global_module.settings_module import set_dir, set_params, set_dict
import time
from global_module.implementation_module import reader
from global_module.implementation_module import model
import numpy as np


def run_epoch(session, eval_op, model_obj, dict_obj, verbose=False):
    print('\nrun epoch')

    output_file = open(set_dir.Directory('TE').log_emb_path + '/word_embedding.csv', 'w')

    params = model_obj.params
    dir_obj = model_obj.dir_obj
    data_filename = dir_obj.data_filename
    label_filename = dir_obj.label_filename

    for step, (input_seq_arr, length_arr, label_arr) \
            in enumerate(reader.data_iterator(params, data_filename, label_filename, model_obj.params.indices, dict_obj)):

        feed_dict = {model_obj.word_input: input_seq_arr,
                     model_obj.seq_length: length_arr,
                     model_obj.label: label_arr}

        emb_matrix, logits, _ = session.run([model_obj.word_emb_matrix,
                                                          model_obj.logits,
                                                          eval_op],
                                                         feed_dict=feed_dict)



        for each_emb in emb_matrix:
            output_file.write(' '.join(str(x) for x in each_emb).strip() + '\n')

        break

    print 'Embedding file written ...'


def get_length(filename):
    print('Reading :', filename)
    data_file = open(filename, 'r')
    count = 0
    for _ in data_file:
        count += 1
    data_file.close()
    return count, np.arange(count)


def init_test():
    mode_train, mode_test = 'TR', 'TE'

    dict_obj = set_dict.Dictionary()

    # train object
    params_train = set_params.ParamsClass(mode=mode_train)
    dir_train = set_dir.Directory(mode_train)
    params_train.num_classes = len(dict_obj.label_dict)

    # test object
    params_test = set_params.ParamsClass(mode=mode_test)
    dir_test = set_dir.Directory(mode_test)
    params_test.num_instances, params_test.indices = get_length(dir_test.data_filename)
    params_test.batch_size = 1
    params_test.num_classes = len(dict_obj.label_dict)

    word_emb_path = dir_train.word_embedding
    word_emb_matrix = np.float32(np.genfromtxt(word_emb_path, delimiter=' '))
    params_train.vocab_size = params_test.vocab_size = len(word_emb_matrix)

    print('***** INITIALIZING TF GRAPH *****')

    session = tf.Session()

    with tf.variable_scope("classifier", reuse=None):
        test_obj = model.DeepAttentionClassifier(params_test, dir_test)

    model_saver = tf.train.Saver()
    print('Loading model ...')
    model_saver.restore(session, set_dir.Directory('TE').test_model)

    print('**** MODEL LOADED ****\n')

    return session, test_obj


def run_test(session, test_obj, dict_obj):
    start_time = time.time()

    print("Starting test computation\n")
    run_epoch(session, tf.no_op(), test_obj, dict_obj)

    curr_time = time.time()
    print('1 epoch run takes ' + str(((curr_time - start_time) / 60)) + ' minutes.')

def main():
    session, test_obj = init_test()
    dict_obj = set_dict.Dictionary()
    run_test(session, test_obj, dict_obj)


if __name__ == "__main__":
    main()