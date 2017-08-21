import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from global_module.settings_module import set_dir
import numpy as np

#TODO: clean code

LOG_DIR = set_dir.Directory('TE').log_emb_path
metadata = LOG_DIR + '/word_metadata.tsv'
wordemb = LOG_DIR + '/word_embedding.csv'
emb = tf.Variable(np.genfromtxt(wordemb), name='word_emb')

with tf.Session() as sess:
    saver = tf.train.Saver([emb])

    sess.run(emb.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'emb.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = emb.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)