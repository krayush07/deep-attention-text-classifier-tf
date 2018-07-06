# [Deep-Attention Text Classifier in Tensorflow](https://krayush07.github.io/deep-attention-text-classifier-tf/)
This repository contains code for text classification using attention mechanism in Tensorflow with tensorboard visualization.

<br/>

# Requirements
* Python 2.7<br/>
* Tensorflow 1.2.1<br/>
* Numpy<br/>

<br/>

# Project Module
* **_[utility_dir](/global_module/utility_dir):_** storage module for data, vocab files, saved models, tensorboard logs, outputs.

* _**[pre_processing_module](/global_module/pre_processing_module):**_ code for pre-processing text file which includes sampling infrequent words, creation of training vocab and classes in form of pickle dictionary.

* **_[implementation_module](/global_module/implementation_module):_** code for model architecture, data reader, training pipeline and test pipeline.

* **_[settings_module](/global_module/settings_module)_**: code to set directory paths (data path, vocab path, model path etc.), set model parameters (hidden dim, attention dim, regularization, dropout etc.), set vocab dictionary.

* **_[run_module](/global_module/run_module):_** wrapper code to execute end-to-end train and test pipeline.

* **_[viz_module](/global_module/viz_module):_** code to generate embedding visualization via tensorboard.

* **_[utility_code](/global_module/utility_code):_** other utility codes

<br/>

# How to run
* **train:** `python -m global_module.run_module.run_train`

* **test:** `python -m global_module.run_module.run_test`

* **visualize tensorboard:** `tensorboard --logdir=PATH-TO-LOG-DIR`

<br/>

# Data Sample
* **[Utterance file](/global_module/utility_dir/folder1/data/raw_tokenized_train.txt)**
    * _it is hard to resist_
    * _But something seems to be missing ._
    * _A movie of technical skill and rare depth of intellect and feeling ._
    * _Brosnan is more feral in this film than I 've seen him before_
    * . . .
    
* **[Utterance label](/global_module/utility_dir/folder1/data/label_train.txt)**
    * _neg_
    * _neg_
    * _pos_
    * _neg_
    * . . .
    
<br/>

# How to change model parameters

Go to `set_params.py` [here](/global_module/settings_module/set_params.py).


<br/>


# Model Graph

![alt text](global_module/utility_dir/viz/model.png?raw=true "model")


# Loss and Accuracy Plots

![alt text](global_module/utility_dir/viz/train_loss.png?raw=true "train_loss")

![alt text](global_module/utility_dir/viz/train_acc.png?raw=true "train_acc")

![alt text](global_module/utility_dir/viz/valid_loss.png?raw=true "valid_loss")

![alt text](global_module/utility_dir/viz/valid_acc.png?raw=true "valid_acc")


#Histogram

![alt text](global_module/utility_dir/viz/histograms.png?raw=true "histogram")

# Embedding Visualization

![alt text](global_module/utility_dir/viz/1999.png?raw=true "1999")

![alt text](global_module/utility_dir/viz/engineer.png?raw=true "engineer")

![alt text](global_module/utility_dir/viz/fever.png?raw=true "fever")

![alt text](global_module/utility_dir/viz/society.png?raw=true "society")

![alt text](global_module/utility_dir/viz/tomato.png?raw=true "tomato")