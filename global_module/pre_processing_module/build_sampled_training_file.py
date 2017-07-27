import cPickle
import re

from global_module.settings_module import set_dir, set_params

glove_dict = cPickle.load(open(set_dir.Directory('TR').glove_path, 'rb'))

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
            string[j] = re.sub(r'#[0-9]+', r'', string[j].strip())
            tokenized_sent = string[j].split(" ")
            tokenized_string = ' '.join(tokenized_sent)
            tokenized_training_string += tokenized_string + '\t'

            for token in tokenized_sent:
                if (word_dict.has_key(token) == False):
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
                words = token.split("#")
                if (word_dict[words[0]] <= threshold):
                    # modified_string += 'UNK' + '#' + words[1] + ' '
                    modified_string += 'UNK' + ' '
                    if (rare_words.has_key(words[0]) == False):
                        rare_words[words[0]] = 1
                        rare_words_count += 1
                elif (config.use_unknown_word == True):
                    modified_string += token + ' '
                elif (glove_dict.has_key(words[0]) == False and config.use_random_initializer == False and config.use_unknown_word == False):
                    # modified_string += 'UNK' + '#' + words[1] + ' '
                    modified_string += 'UNK' + ' '
                    if (rare_words.has_key(words[0]) == False):
                        rare_words[words[0]] = 1
                        rare_words_count += 1
                else:
                    modified_string += token + ' '
            modified_string = modified_string.rstrip() + '\t'
        training_file_pointer.write(modified_string.rstrip() + '\n')

    raw_training_file_pointer.close()
    print(rare_words)
    print('Reading Completed \n ========================== \n Total unique words: %d \n Rare words: %d\n ==========================\n' % (
        len(word_dict), rare_words_count))


def main():
    raw_training_file = set_dir.Directory('TR').raw_train_path
    training_file = set_dir.Directory('TR').data_filename
    sample_train_file(raw_training_file, training_file, 1)


if __name__ == '__main__':
    main()
