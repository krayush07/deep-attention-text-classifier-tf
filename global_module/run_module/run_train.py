from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import global_module.implementation_module.train as train
from global_module.pre_processing_module import build_word_vocab, build_sampled_training_file, generate_label_file
from global_module.settings_module import set_dict


def load_dictionary():
    """
    Utility function to load training vocab files
    :return:
    """
    return set_dict.Dictionary()


def call_train(dict_obj):
    """
    Utility function to execute main training module
    :param dict_obj: dictionary object
    :return: None
    """
    train.run_train(dict_obj)
    return


def train_util():
    """
    Utility function to execute the training pipeline
    :return: None
    """
    build_sampled_training_file.util()
    build_word_vocab.util()
    generate_label_file.util()
    dict_obj = load_dictionary()
    call_train(dict_obj)
    return None


def main():
    """
    Starting module for CLSTM testing
    :return:
    """
    print('STARTING TRAINING')
    train_util()
    return None


if __name__ == '__main__':
    main()
