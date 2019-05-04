# Read the training data and build the vocab (words and sentences)
# Separate out the tokenized training file
# Read the vocab and extract corresponding glove vectors

# id = 0 for padding
# id = 1 for unkown words

# word_vocab.pkl 	 					-> map of unique training words and ids
# glove_present_training_word_vocab.pkl -> map of unique training words that are present in glove data and their new ids
# word_embedding.csv					-> word embedding corresponding to glove_present_words


import pickle
import csv
import pickle
import random
import re

from global_module.settings_module import set_dir, set_params
import collections

dirObj = set_dir.Directory('TR')
dataDir = dirObj.data_path
vocabDir = dirObj.vocab_path
gloveDict = dirObj.glove_path
config = set_params.ParamsClass('TR')


def generate_vocab(training_file):
    word_dict = collections.OrderedDict()
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
            # string[j] = re.sub(r'#[0-9]+', r'', string[j].strip())
            # tokenized_sent = nltk.word_tokenize((string[j]).decode('utf-8'))
            # tokenized_sent.append('<eos>')
            # tokenized_sent = string[j].split(" ")
            tokenized_sent = string[j].split(" ")
            tokenized_string = ' '.join(tokenized_sent)
            tokenized_training_string += tokenized_string + '\t'

            for token in tokenized_sent:
                if (token in word_dict == False):
                    word_dict[token] = word_counter
                    word_counter += 1

        # tokenized_training_file.write(tokenized_training_string.encode('utf-8').rstrip('\t'))
        tokenized_training_file.write(tokenized_training_string.rstrip('\t'))
        tokenized_training_file.write('\n')
        curr_seq_length = len(tokenized_training_string.split())
        if (curr_seq_length > max_sequence_length):
            max_sequence_length = curr_seq_length

    word_vocab = open(set_dir.Directory('TR').word_vocab_dict, 'wb')

    pickle.dump(word_dict, word_vocab)

    word_vocab.close()
    training_file_pointer.close()
    tokenized_training_file.close()

    print('Reading Completed \n ========================== '
          '\n Unique tokens: excluding padding and unkown words %d '
          '\n Max. sequence length: %d'
          '\n ==========================\n'
          % (word_counter - 2, max_sequence_length))

    # print(word_dict)
    return word_dict


def extract_glove_vectors(word_vocab_file, glove_file):
    glove_vocab_dict = pickle.load(open(glove_file, 'rb'))
    word_vocab_dict = pickle.load(open(word_vocab_file, 'rb'))

    length_word_vector = 0

    glove_present_training_word_vocab_dict = collections.OrderedDict()
    glove_present_training_word_counter = 2  # 3
    # glove_present_training_word_counter = 1
    glove_present_word_vector_dict = collections.OrderedDict()

    glove_present_training_word_vocab_dict['PAD'] = 0
    glove_present_training_word_vocab_dict['UNK'] = 1  # 2
    glove_present_word_vector_dict[1] = glove_vocab_dict['food']

    if (length_word_vector == 0):
        length_word_vector = len(glove_vocab_dict['food'].split(' '))

    for key, value in word_vocab_dict.items():

        if(config.all_lowercase):
            if (glove_vocab_dict.has_key(key.lower())):
                key = key.lower()
            elif (glove_vocab_dict.has_key(key)):
                key = key
            elif (glove_vocab_dict.has_key(key.title())):
                key = key.title()
            elif (glove_vocab_dict.has_key(key.upper())):
                key = key.upper()
            else:
                key = key.lower()

        if(key not in glove_present_training_word_vocab_dict):
            if (config.use_unknown_word):
                if (glove_vocab_dict.has_key(key) and config.use_random_initializer == False):
                    if (key != 'UNK'):
                        glove_present_training_word_vocab_dict[key] = glove_present_training_word_counter
                        glove_present_word_vector_dict[glove_present_training_word_counter] = glove_vocab_dict.get(key)
                        glove_present_training_word_counter += 1
                else:
                    glove_present_training_word_vocab_dict[key] = glove_present_training_word_counter
                    vec_str = ''
                    for i in range(length_word_vector):
                        vec_str += str(round(random.uniform(-0.9, 0.9), 6)) + ' '
                    glove_present_word_vector_dict[glove_present_training_word_counter] = vec_str.strip()
                    glove_present_training_word_counter += 1
            elif (glove_vocab_dict.has_key(key) and config.use_random_initializer == False and config.use_unknown_word == False):
                if (key != 'UNK'):
                    glove_present_training_word_vocab_dict[key] = glove_present_training_word_counter
                    glove_present_word_vector_dict[glove_present_training_word_counter] = glove_vocab_dict.get(key)
                    glove_present_training_word_counter += 1
            elif (config.use_random_initializer):
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
    pickle.dump(glove_present_training_word_vocab_dict, glove_present_training_word_vocab)

    print(glove_present_training_word_vocab_dict)

    print('Glove_present_unique_training_tokens, Total unique tokens, Glove token size')
    print(len(glove_present_training_word_vocab_dict), len(word_vocab_dict), len(glove_vocab_dict))

    word_vector_file.close()

    print('\nVocab Size:')
    # print(len(glove_present_word_vector_dict)+2)
    print(len(glove_present_training_word_vocab_dict))

    glove_present_training_word_vocab.close()
    # return(len(glove_present_word_vector_dict)+2)

    #####
    #   WORD METADATA
    ####
    meta_file = open(dirObj.word_emb_tsv, 'w')
    # meta_file.write('Word' + '\t' + 'Id' + '\n')
    for key, value in glove_present_training_word_vocab_dict.items():
        # meta_file.write(key + '\t' + str(value) + '\n')
        meta_file.write(key + '\n')
    meta_file.close()
    #####

    return len(glove_present_word_vector_dict) + 1


def util():
    training_file = set_dir.Directory('TR').data_filename
    generate_vocab(training_file)
    vocab_size = extract_glove_vectors(set_dir.Directory('TR').word_vocab_dict, gloveDict)
    return vocab_size

def main():
    training_file = set_dir.Directory('TR').data_filename
    word_dict = generate_vocab(training_file)
    vocab_size = extract_glove_vectors(set_dir.Directory('TR').word_vocab_dict, gloveDict)
    return vocab_size


# extractGloveVectors('glove.300.txt')


if __name__ == '__main__':
    main()
