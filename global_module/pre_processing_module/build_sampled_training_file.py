import pickle
import re

from global_module.settings_module import set_dir, set_params

# glove_dict = pickle.load(open(set_dir.Directory('TR').glove_path, 'rb'))
glove_dict = pickle.load(open('/Users/ayushobserve/Downloads/w2v_embedding_new_dict.pkl', 'rb'))
config = set_params.ParamsClass('TR')


def sample_train_file(raw_training_file, training_file, threshold):
    raw_training_file_pointer = open(raw_training_file, 'r')
    training_file_pointer = open(training_file, 'w')
    word_dict = {}

    print('\nReading raw training file .... ')

    for line in raw_training_file_pointer:
        line = line.rstrip()
        # line = line.lower()
        string = re.split(r'\t', line)
        size = len(string)
        tokenized_training_string = ''
        for j in range(size):
            # string[j] = re.sub(r'#[0-9]+', r'', string[j].strip())
            tokenized_sent = string[j].split(" ")
            tokenized_string = ' '.join(tokenized_sent)
            tokenized_training_string += tokenized_string + '\t'

            for token in tokenized_sent:
                if token not in word_dict:
                    word_dict[token] = 1
                else:
                    word_dict[token] += word_dict[token] + 1

    raw_training_file_pointer.seek(0)

    rare_words_count = 0
    rare_words = {}
    for line in raw_training_file_pointer:
        line = line.strip()
        # line = line.lower()
        string = line.split("\t")
        modified_string = ''
        for utterance in string:
            utt = utterance.split(" ")
            for token in utt:
                # if(config.use_random_initializer):
                #     modified_string += token + ' '
                # else:
                # words = token.split("#")
                if word_dict[token] <= threshold:
                    # modified_string += 'UNK' + '#' + words[1] + ' '
                    modified_string += 'UNK' + ' '
                    if token not in rare_words:
                        rare_words[token] = 1
                        rare_words_count += 1
                elif config.use_unknown_word:
                    modified_string += token + ' '
                elif token not in glove_dict and not config.use_random_initializer and not config.use_unknown_word:
                    # modified_string += 'UNK' + '#' + words[1] + ' '
                    modified_string += 'UNK' + ' '
                    if token not in rare_words:
                        rare_words[token] = 1
                        rare_words_count += 1
                else:
                    modified_string += token + ' '
            modified_string = modified_string.rstrip() + '\t'
        training_file_pointer.write(modified_string.rstrip() + '\n')

    raw_training_file_pointer.close()
    print(rare_words)
    print('Reading Completed \n ========================== \n Total unique words: %d \n Rare words: %d\n ==========================\n' % (
        len(word_dict), rare_words_count))


def util():
    raw_training_file = set_dir.Directory('TR').raw_train_path
    training_file = set_dir.Directory('TR').data_filename
    sample_train_file(raw_training_file, training_file, set_params.ParamsClass().sampling_threshold)

# def main():
#     raw_training_file = set_dir.Directory('TR').raw_train_path
#     training_file = set_dir.Directory('TR').data_filename
#     sample_train_file(raw_training_file, training_file, set_params.ParamsClass().sampling_threshold)


# if __name__ == '__main__':
#     main()
