import cPickle as pickle
from global_module.settings_module import set_dir

rel_dir = set_dir.Directory('TR')


def convert(test_filename):
    label_dict = pickle.load(open(rel_dir.label_map_dict, 'rb'))
    test_file = open(test_filename, 'r')
    op_file = open(test_filename + '_output.txt', 'w')

    new_map = {}

    for actual_id, mapped_id in label_dict.iteritems():
        new_map[mapped_id] = actual_id

    for line in test_file:
        line = line.strip()
        op_file.write(new_map[int(line) - 1] + '\n')

    op_file.close()
    test_file.close()

    # convert('/home/aykumar/aykumar_home/self/deep-text-classifier/global_module/utility_dir/folder1/output/dummy_rnn.txt')
