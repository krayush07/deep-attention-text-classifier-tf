import numpy as np

from global_module.settings_module import set_params, set_dict


def get_index_string(utt, word_dict, params):
    index_string = ''
    for each_token in utt.split():
        if params.all_lowercase:
            if each_token.lower() in word_dict:
                each_token = each_token.lower()
            elif each_token in word_dict:
                each_token = each_token
            elif each_token.title() in word_dict:
                each_token = each_token.title()
            elif each_token.upper() in word_dict:
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
    global curr_label
    data_file_arr = open(data_filename, 'r').readlines()
    label_file_arr = open(label_filename, 'r').readlines()

    input_seq_arr = []
    length_arr = []
    label_arr = []

    label_not_present = 0

    for each_idx in index_arr:
        curr_line = data_file_arr[each_idx].strip()
        if  dict_obj.label_dict.has_key(label_file_arr[each_idx].strip()):
            curr_label = dict_obj.label_dict[label_file_arr[each_idx].strip()]
        elif params.mode == 'TE':
            curr_label = '-1'
            label_not_present += 1

        curr_seq_token_len, curr_seq_token_id = get_index_string(curr_line, dict_obj.word_dict, params)
        curr_seq_token_string = format_string(curr_seq_token_id, curr_seq_token_len, params.MAX_SEQ_LEN)

        if curr_seq_token_len > params.MAX_SEQ_LEN:
            curr_seq_token_len = params.MAX_SEQ_LEN

        input_seq_arr.append(curr_seq_token_string)
        length_arr.append(curr_seq_token_len)
        label_arr.append(curr_label)

    print('Reading: DONE, Label not present: %d' % label_not_present)
    return input_seq_arr, length_arr, label_arr


def data_iterator(params, data_filename, label_filename, index_arr, dict_obj):
    input_seq_arr, length_arr, label_arr = generate_id_map(params, data_filename, label_filename, index_arr, dict_obj)

    batch_size = params.batch_size
    num_batches = len(index_arr) / params.batch_size

    for i in range(num_batches):
        curr_input_seq_arr = np.loadtxt(input_seq_arr[i * batch_size: (i + 1) * batch_size], dtype=np.int32)
        if batch_size == 1:
            curr_input_seq_arr = np.expand_dims(curr_input_seq_arr, axis=0)

        curr_length_arr = np.array(length_arr[i * batch_size: (i + 1) * batch_size], dtype=np.int32)
        curr_label_arr = np.array(label_arr[i * batch_size: (i + 1) * batch_size], dtype=np.int32)

        yield (curr_input_seq_arr, curr_length_arr, curr_label_arr)
        # print("A")


def get_length(filename):
    print('Reading :', filename)
    data_file = open(filename, 'r')
    count = 0
    for line in data_file:
        count += 1
    data_file.close()
    return count, np.arange(count)
