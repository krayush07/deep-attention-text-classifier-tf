import cPickle as pickle
from global_module.settings_module import set_dir


def generate_indexed_labels():
    label_hash = {}
    input_file = open(set_dir.Directory('TR').label_filename).readlines()
    curr_count = 0
    for each_label in input_file:
        curr_label = each_label.strip()
        if curr_label not in label_hash:
            label_hash[curr_label] = curr_count
            curr_count += 1

    label_map_file = open(set_dir.Directory('TR').label_map_dict, 'wb')
    pickle.dump(label_hash, label_map_file, protocol=pickle.HIGHEST_PROTOCOL)

    print 'Total classes %d' % (len(label_hash))


def main():
    generate_indexed_labels()


if __name__ == '__main__':
    main()
