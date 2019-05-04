import pickle

import numpy as np

from global_module.settings_module import set_dir


class Dictionary:
    def __init__(self, mode='TR'):
        """
        :param mode: 'TR' for train, 'TE' for test, 'VA' for valid
        """
        self.mode = mode
        self.rel_dir = set_dir.Directory(mode)
        # gloveDict = rel_dir.glove_path
        self.word_dict = pickle.load(open(self.rel_dir.glove_present_training_word_vocab, 'rb'))
        self.word_emb = self.rel_dir.word_embedding
        self.glove_present_word_csv = np.float32(np.genfromtxt(self.word_emb, delimiter=' '))
        self.label_dict = pickle.load(open(self.rel_dir.label_map_dict, 'rb'))
