import tensorflow as tf
import numpy as np
from CycleGAN import CycleGAN

flags = tf.app.flags
# DATA SETTINGS (INPUT & OUTPUT)
flags.DEFINE_string("data_dir", "./datasets", "Directory of data [./datasets]")
flags.DEFINE_string("dataset", "monet2photo", "The dataset would be used [monet2photo]")
flags.DEFINE_boolean("is_grayscale", False, "Turn the images into grayscale or not [False]")
flags.DEFINE_boolean("is_crop", True, "Crop the images (if input_size > output_size) or not [True]")
flags.DEFINE_string("download_script", "./download_dataset.sh", "The script for downloading dataset [download_dataset.sh]")
# MODEL STRUCTURE SETTINGS
## GLOBAL
flags.DEFINE_integer("fc_dim", 64, "The count of filters of first layer of convolution network [64]")
flags.DEFINE_integer("fd_dim", 64, "The count of filters of first layer of deconvolution network [64]")
# TRAINING SETTINGS
flags.DEFINE_integer("iterations", 50000, "Number iterations to train [50000]")
flags.DEFINE_integer("batch_size", 1, "Batch size for training [1]")
flags.DEFINE_integer("sample_num", 1, "Number of sampling [1]")
flags.DEFINE_float("d_learning_rate", 0.0002, "Learning rate for discriminator [0.0002]")
flags.DEFINE_float("g_learning_rate", 0.0002, "Learning rate for generator [0.0002]")
flags.DEFINE_float("L1_lambda", 10, "Scale of consistency loss [10]")
flags.DEFINE_float("identity_loss_scale", 0., "Scale of identity loss [0]")
flags.DEFINE_integer("pool_size", 50, "The size of images pool [50]")
## ADAM (FOR NOT W-GAN)
flags.DEFINE_float("beta1", 0.9, "Momentum parameter for Adam [0.9]")
# LOG AND MODEL
flags.DEFINE_string("checkpoint_dir", "./model", "Directory for pre-loading model [./model]")
flags.DEFINE_string("save_dir", "./model", "Directory for saving model [./model]")
flags.DEFINE_string("images_dir", "./images", "Directory for sampled images [./images]")
flags.DEFINE_string("training_log", "./train.log", "Path of training log [./train.log]")
flags.DEFINE_string("testing_log", "./test.log", "Path of testing log [./test.log]")
flags.DEFINE_integer("save_step", 2500, "save the model every N step [2500]")
flags.DEFINE_integer("sample_step", 1000, "sample every N step [1000]")
flags.DEFINE_integer("verbose_step", 100, "output log every N step [100]")
# OTHER
flags.DEFINE_boolean("is_train", True, "Training or testing [True]")
FLAGS = flags.FLAGS

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

with tf.Session(config=run_config) as sess:
    model = CycleGAN(sess, FLAGS)
    if FLAGS.is_train:
        model.train(FLAGS)
    else:
        model.test()
