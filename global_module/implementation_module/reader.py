import cPickle
import cPickle as pickle
import re

import numpy as np
from global_module.settings_module import set_params, set_dir, set_dict


def get_index_string(utt, word_dict):
    index_string = ''
    for each_token in utt.split():

        if (word_dict.has_key(each_token.lower())):
            each_token = each_token.lower()
        elif (word_dict.has_key(each_token)):
            each_token = each_token
        elif (word_dict.has_key(each_token.title())):
            each_token = each_token.title()
        elif (word_dict.has_key(each_token.upper())):
            each_token = each_token.upper()
        else:
            each_token = each_token.lower()

        index_string += str(word_dict.get(each_token, word_dict.get("UNK"))) + '\t'
    return len(index_string.strip().split()), index_string.strip()


def pad_string(id_string, curr_len, max_seq_len):
    id_string = id_string.strip() + '\t'
    while curr_len < max_seq_len:
        id_string += '0\t'
        curr_len += 1
    return id_string.strip()


def add_dummy_context_string(curr_context_string, curr_num_context, max_num_context, indiv_max_seq_len):
    for i in range(curr_num_context, max_num_context):
        context_string = ''
        for j in range(indiv_max_seq_len):
            context_string += '0\t'
        curr_context_string.append(context_string.strip())
    return curr_context_string


def strip_extra_sequence(feature_id_string_arr, max_seq_len):
    for i in range(len(feature_id_string_arr)):
        string_id_split = feature_id_string_arr[i].split('\t')[:max_seq_len]
        string_id_string = '\t'.join(string_id_split)
        feature_id_string_arr[i] = string_id_string.strip()
    return feature_id_string_arr


def format_string(inp_string, curr_string_len, max_len):
    if curr_string_len > max_len:
        print('Maximum SEQ LENGTH reached. Stripping extra sequence.\n')
        op_string = '\t'.join(inp_string.split('\t')[:max_len])
    else:
        op_string = pad_string(inp_string, curr_string_len, max_len)
    return op_string


def generate_id_map(params, data_filename, label_filename, index_arr, dict_obj):
    data_file_arr = open(data_filename, 'r').readlines()
    label_file_arr = open(label_filename, 'r').readlines()

    input_seq_arr = []
    label_arr = []

    for each_idx in index_arr:
        curr_line = data_file_arr[each_idx].strip()
        curr_label = dict_obj.label_dict[label_file_arr[each_idx].strip()]

        curr_seq_token_len, curr_seq_token_id = get_index_string(curr_line, dict_obj.word_dict)
        curr_seq_token_string = format_string(curr_seq_token_id, curr_seq_token_len, params.MAX_SEQ_LEN)

        input_seq_arr.append(curr_seq_token_string)
        label_arr.append(curr_label)

    print('Reading: DONE')
    return input_seq_arr, label_arr


def data_iterator(params, data_filename, label_filename, index_arr, dict_obj):
    input_seq_arr, label_arr = generate_id_map(params, data_filename, label_filename, index_arr, dict_obj)

    batch_size = params.batch_size
    num_batches = len(index_arr) / params.batch_size

    for i in range(num_batches):
        curr_input_seq_arr = np.loadtxt(input_seq_arr[i * batch_size: (i + 1) * batch_size], dtype=np.int32)
        if (batch_size == 1):
            curr_input_seq_arr = np.expand_dims(curr_input_seq_arr, axis=0)

        curr_label_arr = np.array(label_arr[i * batch_size: (i + 1) * batch_size], dtype=np.int32)

        yield (curr_input_seq_arr, curr_label_arr)
        # print("A")


def getLength(fileName):
    print('Reading :', fileName)
    dataFile = open(fileName, 'r')
    count = 0
    for line in dataFile:
        count += 1
    dataFile.close()
    return count, np.arange(count)


def main():
    data_file = '/home/aykumar/aykumar_home/aykumar_dir/multi_view/global_module/utility_dir/folder1/data/raw_tokenized_train.txt'
    label_file = '/home/aykumar/aykumar_home/aykumar_dir/multi_view/global_module/utility_dir/folder1/data/labels_train.txt'
    count, utterance_idx = getLength(data_file)
    dictObj = set_dict.Dictionary('TR')
    config = set_params.ParamsClass('TR')
    flag = 'TR'

    a, b, c, d, e = data_iterator(config, data_file, label_file, utterance_idx, dictObj)

    # for step, (a, b, c, d, e, f, g, h) in enumerate(data_iterator(config, data_file, label_file, utterance_idx, dictObj)):
    #     print 1
    # np.savetxt('a.txt', a, fmt='%d')
    # np.savetxt('b.txt', b, fmt='%d')
    # np.savetxt('c.txt', c, fmt='%d')
    # np.savetxt('d.txt', d, fmt='%d')
    # np.savetxt('e.txt', e, fmt='%d')
    # np.savetxt('f.txt', f, fmt='%d')
    # print g
    # np.savetxt('g.txt', np.squeeze(g), fmt='%d')
    # np.savetxt('h.txt', h)


if __name__ == '__main__':
    main()
