# Read the training data and build the vocab (words and sentences)
# Separate out the tokenized training file
# Read the vocab and extract corresponding glove vectors

# id = 0 for padding
# id = 1 for unkown words

# word_vocab.pkl 	 					-> map of unique training words and ids
# glove_present_training_word_vocab.pkl -> map of unique training words that are present in glove data and their new ids
# word_embedding.csv					-> word embedding corresponding to glove_present_words


import cPickle
import csv
import pickle
import random
import re

from global_module.settings_module import set_dir, set_params

dataDir = set_dir.Directory('TR').data_path
vocabDir = set_dir.Directory('TR').vocab_path
gloveDict = set_dir.Directory('TR').glove_path
config = set_params.ParamsClass('TR')


def generate_vocab(training_file):
    word_dict = {}
    word_counter = 2
    max_sequence_length = 0

    training_file_pointer = open(training_file, 'r')

    print('\nReading Training File .... ')

    tokenized_training_file = open(dataDir + '/tokenized_training', 'w')

    for line in training_file_pointer:
        line = line.rstrip()
        string = re.split(r'\t', line)
        size = len(string)
        tokenized_training_string = ''
        for j in range(size):
            string[j] = re.sub(r'#[0-9]+', r'', string[j].strip())
            # tokenized_sent = nltk.word_tokenize((string[j]).decode('utf-8'))
            # tokenized_sent.append('<eos>')
            # tokenized_sent = string[j].split(" ")
            tokenized_sent = string[j].split(" ")
            tokenized_string = ' '.join(tokenized_sent)
            tokenized_training_string += tokenized_string + '\t'

            for token in tokenized_sent:
                if token not in word_dict:
                    word_dict[token] = word_counter
                    word_counter += 1

        # tokenized_training_file.write(tokenized_training_string.encode('utf-8').rstrip('\t'))
        tokenized_training_file.write(tokenized_training_string.rstrip('\t'))
        tokenized_training_file.write('\n')
        curr_seq_length = len(tokenized_training_string.split())
        if curr_seq_length > max_sequence_length:
            max_sequence_length = curr_seq_length

    word_vocab = open(set_dir.Directory('TR').word_vocab_dict, 'wb')

    pickle.dump(word_dict, word_vocab, protocol=cPickle.HIGHEST_PROTOCOL)

    word_vocab.close()
    training_file_pointer.close()
    tokenized_training_file.close()

    print(
        'Reading Completed \n ========================== \n Unique tokens: excluding padding and unkown words %d \n Max. sequence length: %d\n ==========================\n' % (
            word_counter - 2, max_sequence_length))

    # print(word_dict)
    return word_dict


def extract_glove_vectors(word_vocab_file, glove_file):
    glove_vocab_dict = cPickle.load(open(glove_file, 'rb'))
    word_vocab_dict = cPickle.load(open(word_vocab_file, 'rb'))

    length_word_vector = 0

    glove_present_training_word_vocab_dict = {}
    glove_present_training_word_counter = 2  # 3
    # glove_present_training_word_counter = 1
    glove_present_word_vector_dict = {}

    glove_present_training_word_vocab_dict['UNK'] = 1  # 2
    glove_present_word_vector_dict[1] = glove_vocab_dict.get('UNK')

    if length_word_vector == 0:
        length_word_vector = len(glove_vocab_dict.get('the').split(' '))

    for key, value in word_vocab_dict.items():

        if (config.all_lowercase):
            if key.lower() in glove_vocab_dict:
                key = key.lower()
            elif key in glove_vocab_dict:
                key = key
            elif key.title() in glove_vocab_dict:
                key = key.title()
            elif key.upper() in glove_vocab_dict:
                key = key.upper()
            else:
                key = key.lower()

        if key not in glove_present_training_word_vocab_dict:
            if config.use_unknown_word:
                if key in glove_vocab_dict and not config.use_random_initializer:
                    if key != 'UNK':
                        glove_present_training_word_vocab_dict[key] = glove_present_training_word_counter
                        glove_present_word_vector_dict[glove_present_training_word_counter] = glove_vocab_dict.get(key)
                        glove_present_training_word_counter += 1
                else:
                    glove_present_training_word_vocab_dict[key] = glove_present_training_word_counter
                    vec_str = ''
                    for i in range(length_word_vector):
                        vec_str += str(round(random.uniform(-0.1, 0.1), 6)) + ' '
                    glove_present_word_vector_dict[glove_present_training_word_counter] = vec_str.strip()
                    glove_present_training_word_counter += 1
            elif key in glove_vocab_dict and not config.use_random_initializer and not config.use_unknown_word:
                if key != 'UNK':
                    glove_present_training_word_vocab_dict[key] = glove_present_training_word_counter
                    glove_present_word_vector_dict[glove_present_training_word_counter] = glove_vocab_dict.get(key)
                    glove_present_training_word_counter += 1
            elif config.use_random_initializer:
                glove_present_training_word_vocab_dict[key] = glove_present_training_word_counter
                glove_present_word_vector_dict[glove_present_training_word_counter] = glove_vocab_dict.get('UNK')
                glove_present_training_word_counter += 1
                # else :
                #     print('Error')

    word_vector_file = open(set_dir.Directory('TR').word_embedding, 'w')
    writer = csv.writer(word_vector_file)
    string = ''
    for i in range(length_word_vector):
        string += '0 '
    word_vector_file.write(string.rstrip(' ') + '\n')
    # word_vector_file.write(string.rstrip(' ') + '\n') # zeros vector (id 1)
    for key, value in glove_present_word_vector_dict.items():
        writer.writerow([value])

    glove_present_training_word_vocab = open(set_dir.Directory('TR').glove_present_training_word_vocab, 'wb')
    pickle.dump(glove_present_training_word_vocab_dict, glove_present_training_word_vocab, protocol=cPickle.HIGHEST_PROTOCOL)

    print(glove_present_training_word_vocab_dict)

    print('Glove_present_unique_training_tokens, Total unique tokens, Glove token size')
    print(len(glove_present_word_vector_dict), len(word_vocab_dict), len(glove_vocab_dict))

    word_vector_file.close()

    print('\nVocab Size:')
    # print(len(glove_present_word_vector_dict)+2)
    print(len(glove_present_word_vector_dict) + 1)

    glove_present_training_word_vocab.close()
    # return(len(glove_present_word_vector_dict)+2)
    return len(glove_present_word_vector_dict) + 1


def main():
    training_file = set_dir.Directory('TR').data_filename
    word_dict = generate_vocab(training_file)
    vocab_size = extract_glove_vectors(set_dir.Directory('TR').word_vocab_dict, gloveDict)
    return vocab_size


# extractGloveVectors('glove.300.txt')


if __name__ == '__main__':
    main()
