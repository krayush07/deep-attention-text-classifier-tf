from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from global_module.settings_module import set_dict
import global_module.implementation_module.test as test_model
from global_module.utility_code import convert_pred_to_class as convert


#########################################################
# Utility function to load training vocab files
#########################################################

def load_dictionary():
    """
    Utility function to load training vocab files
    :return:
    """
    return set_dict.Dictionary('TE')


def initialize_test_session():
    dict_obj = test_util()
    session, mtest = test_model.init_test()
    return session, mtest, dict_obj


def call_test(session, mtest, dict_obj):
    test_model.run_test(session, mtest, dict_obj)


def test_util():
    """
    Utility function to execute the testing pipeline
    :return:
    """
    dict_obj = load_dictionary()
    return dict_obj


def main():
    """
    Starting module for testing
    :return:
    """
    print('STARTING TESTING')
    session, mtest, dict_obj = initialize_test_session()
    call_test(session, mtest, dict_obj)
    convert.convert(dict_obj.rel_dir.test_cost_path)


if __name__ == '__main__':
    main()
