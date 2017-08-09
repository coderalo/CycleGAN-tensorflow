import os
import sys
import glob
import json
import numpy as np
import subprocess
from math import ceil, floor, log
from utils import *
from model_utils import *
from data_utils import *
from ImagePool import ImagePool
from scipy.misc import imsave

class CycleGAN:
        
    ########################################################
    #            initialize and main training              #
    ########################################################    
    
    def __init__(self, sess, FLAGS):
        self.sess = sess
        self.input_height, self.input_width, self.output_height, self.output_width = \
                valid_input_and_output((FLAGS.input_height, FLAGS.input_width), 
                        (FLAGS.output_height, FLAGS.output_width))
        self.input_shape = (self.input_height, self.input_width)
        self.output_shape = (self.output_height, self.output_width)
        self.channels = FLAGS.channels
        self.crop = FLAGS.is_crop
        self.fc_dim = FLAGS.fc_dim
        self.fd_dim = FLAGS.fd_dim
        self.batch_size = FLAGS.batch_size
        self.sample_num = FLAGS.sample_num
        self.save_step = FLAGS.save_step
        self.sample_step = FLAGS.sample_step
        self.verbose_step = FLAGS.verbose_step
        self.checkpoint_dir = check_dir(FLAGS.checkpoint_dir)
        self.save_dir = check_dir(FLAGS.save_dir)
        self.images_dir = check_dir(FLAGS.images_dir)
        self.data_dir = FLAGS.data_dir
        self.dataset = FLAGS.dataset
        self.L1_lambda = FLAGS.L1_lambda
        self.identity_loss_scale = FLAGS.identity_loss_scale
        self.pool_size = FLAGS.pool_size
        self.pool_A, self.pool_B = ImagePool(self.pool_size), ImagePool(self.pool_size)
        self.is_train = FLAGS.is_train
        if self.is_train == True:
            self.training_log = check_log(FLAGS.training_log)
        self.download_script = FLAGS.download_script

        self.prepare_data()
        self.build_model()

    def train(self, config):
        self.DA_optimizer = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta1) \
                .minimize(self.DA_loss, var_list=self.da_vars)
        self.DB_optimizer = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta1) \
                .minimize(self.DB_loss, var_list=self.db_vars)
        self.GA2B_optimizer = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.beta1) \
                .minimize(self.GA2B_loss, var_list=self.ga2b_vars)
        self.GB2A_optimizer = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.beta1) \
                .minimize(self.GB2A_loss, var_list=self.gb2a_vars)
        tf.global_variables_initializer().run()

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        counter = 1
        print_time_info("Start training...")
        checker, before_counter = self.load_model()
        if checker: counter = before_counter
        
        self.errD_list, self.errG_list = [], []
        for __ in range(config.iterations):
            self.train_batch(counter)
            if counter % self.sample_step == 0: self.sample_test(counter)
            if counter % self.save_step == 0: self.save_model(counter)
            counter += 1

    ########################################################
    #                    model structure                   #
    ########################################################    
    
    def build_model(self):
        
        self.real_A = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, self.channels], name="real_A")
        self.real_B = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, self.channels], name="real_B")

        self.fake_A = self.generator(self.real_B, reuse=False, name="generatorB2A")
        self.fake_B = self.generator(self.real_A, reuse=False, name="generatorA2B")
        self.cycle_A = self.generator(self.fake_B, reuse=True, name="generatorB2A")
        self.cycle_B = self.generator(self.fake_A, reuse=True, name="generatorA2B")

        self.DA_fake = self.discriminator(self.fake_A, reuse=False, name="discriminatorA")
        self.DB_fake = self.discriminator(self.fake_B, reuse=False, name="discriminatorB")
        
        cycle_loss_A = self.L1_lambda * tf.reduce_mean(tf.abs(self.cycle_A - self.real_A))
        cycle_loss_B = self.L1_lambda * tf.reduce_mean(tf.abs(self.cycle_B - self.real_B))
        cycle_loss = cycle_loss_A + cycle_loss_B
        adv_loss_A2B = tf.reduce_mean(tf.square(self.DB_fake - tf.ones_like(self.DB_fake)))
        adv_loss_B2A = tf.reduce_mean(tf.square(self.DA_fake - tf.ones_like(self.DA_fake)))
        identity_loss_A2B = self.identity_loss_scale * tf.reduce_mean(tf.abs(self.fake_B - self.real_A))
        identity_loss_B2A = self.identity_loss_scale * tf.reduce_mean(tf.abs(self.fake_A - self.real_B))
        
        self.GA2B_loss = cycle_loss + adv_loss_A2B + identity_loss_A2B
        self.GB2A_loss = cycle_loss + adv_loss_B2A + identity_loss_B2A

        self.sample_A = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, self.channels], name="A_sample")
        self.sample_B = tf.placeholder(tf.float32, [self.batch_size, self.input_height, self.input_width, self.channels], name="B_sample")
    
        self.DA_real = self.discriminator(self.real_A, reuse=True, name="discriminatorA")
        self.DB_real = self.discriminator(self.real_B, reuse=True, name="discriminatorB")
        self.DA_sample = self.discriminator(self.sample_A, reuse=True, name="discriminatorA")
        self.DB_sample = self.discriminator(self.sample_B, reuse=True, name="discriminatorB")

        self.DA_real_loss = tf.reduce_mean(tf.square(self.DA_real - tf.ones_like(self.DA_real)))
        self.DB_real_loss = tf.reduce_mean(tf.square(self.DB_real - tf.ones_like(self.DB_real)))
        self.DA_sample_loss = tf.reduce_mean(tf.square(self.DA_sample - tf.zeros_like(self.DA_sample)))
        self.DB_sample_loss = tf.reduce_mean(tf.square(self.DB_sample - tf.zeros_like(self.DB_sample)))
        self.DA_loss = 0.5 * (self.DA_real_loss + self.DA_sample_loss)
        self.DB_loss = 0.5 * (self.DB_real_loss + self.DB_sample_loss)

        self.GA2B_sum = tf.summary.scalar("GA2B_loss", self.GA2B_loss)
        self.GB2A_sum = tf.summary.scalar("GB2A_loss", self.GB2A_loss)
        self.DA_real_sum = tf.summary.scalar("DA_real_loss", self.DA_real_loss)
        self.DA_fake_sum = tf.summary.scalar("DA_fake_loss", self.DA_sample_loss)
        self.DA_sum = tf.summary.scalar("DA_loss", self.DA_loss)
        self.DB_real_sum = tf.summary.scalar("DB_real_loss", self.DB_real_loss)
        self.DB_fake_sum = tf.summary.scalar("DB_fake_loss", self.DB_sample_loss)
        self.DB_sum = tf.summary.scalar("DB_loss", self.DB_loss)
        
        self.DA_sum = tf.summary.merge([self.DA_real_sum, self.DA_fake_sum, self.DA_sum])
        self.DB_sum = tf.summary.merge([self.DB_real_sum, self.DB_fake_sum, self.DB_sum])
        
        t_vars = tf.trainable_variables()

        self.da_vars = [var for var in t_vars if 'discriminatorA' in var.name]
        self.db_vars = [var for var in t_vars if 'discriminatorB' in var.name]
        self.ga2b_vars = [var for var in t_vars if 'generatorA2B' in var.name]
        self.gb2a_vars = [var for var in t_vars if 'generatorB2A' in var.name]

        self.saver = tf.train.Saver()

    def discriminator(self, input_tensor, reuse=False, name="discriminator"):
        with tf.variable_scope(name) as scope:
            if reuse: scope.reuse_variables()
            
            h0 = LeakyReLU(conv2d(input_tensor, self.fc_dim, name="d_h0_conv"))
            h1 = LeakyReLU(instance_norm(conv2d(h0, self.fc_dim * 2, name="d_h1_conv"), "d_in1"))
            h2 = LeakyReLU(instance_norm(conv2d(h1, self.fc_dim * 4, name="d_h2_conv"), "d_in2"))
            h3 = LeakyReLU(instance_norm(conv2d(h2, self.fc_dim * 8, strides=(1, 1), name="d_h3_conv"), "d_in3"))
            h4 = LeakyReLU(conv2d(h3, 1, strides=(1, 1), name="d_h4_conv"))
            
            return h4

    def generator(self, input_tensor, reuse=False, name="generator"):
        with tf.variable_scope(name) as scope:
            if reuse: scope.reuse_variables()
            c0 = tf.pad(input_tensor, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            c1 = tf.nn.relu(instance_norm(conv2d(c0, self.fd_dim, 7, 1, padding='VALID', name='g_e1_conv'), 'g_e1_in'))
            c2 = tf.nn.relu(instance_norm(conv2d(c1, self.fd_dim*2, 3, 2, name='g_e2_conv'), 'g_e2_in'))
            c3 = tf.nn.relu(instance_norm(conv2d(c2, self.fd_dim*4, 3, 2, name='g_e3_conv'), 'g_e3_in'))
            # define G network with 9 resnet blocks
            r1 = residule_block(c3, self.fd_dim*4, name='g_r1')
            r2 = residule_block(r1, self.fd_dim*4, name='g_r2')
            r3 = residule_block(r2, self.fd_dim*4, name='g_r3')
            r4 = residule_block(r3, self.fd_dim*4, name='g_r4')
            r5 = residule_block(r4, self.fd_dim*4, name='g_r5')
            r6 = residule_block(r5, self.fd_dim*4, name='g_r6')
            r7 = residule_block(r6, self.fd_dim*4, name='g_r7')
            r8 = residule_block(r7, self.fd_dim*4, name='g_r8')
            r9 = residule_block(r8, self.fd_dim*4, name='g_r9')
            d1 = deconv2d(r9, (self.batch_size, self.output_shape[0]//2, self.output_shape[1]//2, self.fd_dim*2), 3, 2, name='g_d1_dc')
            d1 = tf.nn.relu(instance_norm(d1, 'g_d1_in'))
            d2 = deconv2d(d1, (self.batch_size, self.output_shape[0], self.output_shape[1], self.fd_dim), 3, 2, name='g_d2_dc')
            d2 = tf.nn.relu(instance_norm(d2, 'g_d2_in'))
            d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
            pred = conv2d(d2, self.channels, 7, 1, padding='VALID', name='g_pred_c')
            return (tf.nn.tanh(pred)/2. + 0.5)

    ########################################################
    #                   train and sample                   #
    ########################################################    

    def train_batch(self, counter):
        batch_A, batch_B = self.get_batch(self.batch_size, is_random=False)
        # Forward G
        fake_A, fake_B = self.sess.run([self.fake_A, self.fake_B], feed_dict={self.real_A: batch_A, self.real_B: batch_B})
        sample_A = self.pool_A(fake_A)
        sample_B = self.pool_B(fake_B)
        # Training
        _, summary_str = self.sess.run([self.GA2B_optimizer, self.GA2B_sum], feed_dict={self.real_A: batch_A, self.real_B: batch_B})
        self.writer.add_summary(summary_str, counter)
        _, summary_str = self.sess.run([self.DB_optimizer, self.DB_sum], feed_dict={self.real_B: batch_B, self.sample_B: sample_B})
        self.writer.add_summary(summary_str, counter)
        _, summary_str = self.sess.run([self.GB2A_optimizer, self.GB2A_sum], feed_dict={self.real_A: batch_A, self.real_B: batch_B})
        self.writer.add_summary(summary_str, counter)
        _, summary_str = self.sess.run([self.DA_optimizer, self.DA_sum], feed_dict={self.real_A: batch_A, self.sample_A: sample_A})
        self.writer.add_summary(summary_str, counter)

        errDA = self.DA_loss.eval({self.real_A: batch_A, self.real_B: batch_B, self.sample_A: sample_A})
        errDB = self.DB_loss.eval({self.real_B: batch_A, self.real_B: batch_B, self.sample_B: sample_B})
        errGA2B = self.GA2B_loss.eval({self.real_A: batch_A, self.real_B: batch_B})
        errGB2A = self.GB2A_loss.eval({self.real_A: batch_A, self.real_B: batch_B})
        
        errD = errDA + errDB
        errG = errGA2B + errGB2A

        if counter % self.verbose_step == 0:
            print_time_info("Iteration {:0>7} errD: {}, errG: {}".format(counter, errD, errG))
            with open(self.training_log, 'a') as file:
                file.write("{},{},{}\n".format(counter, errD, errG))

            self.errD_list.append(errD)
            self.errG_list.append(errG)
    
    def sample_test(self, counter):
        batch_A, batch_B = self.get_batch(self.batch_size, is_random=True)
        fake_A, fake_B = self.sess.run([self.fake_A, self.fake_B], feed_dict={self.real_A: batch_A, self.real_B: batch_B})
        fake_A, fake_B = fake_A[0], fake_B[0]
        image_real_A_path = os.path.join(self.images_dir, "{}_real_A.jpg".format(counter))
        image_real_B_path = os.path.join(self.images_dir, "{}_real_B.jpg".format(counter))
        image_fake_A_path = os.path.join(self.images_dir, "{}_fake_A.jpg".format(counter))
        image_fake_B_path = os.path.join(self.images_dir, "{}_fake_B.jpg".format(counter))
        imsave(image_real_A_path, np.squeeze(batch_A))
        imsave(image_real_B_path, np.squeeze(batch_B))
        imsave(image_fake_A_path, np.squeeze(fake_A))
        imsave(image_fake_B_path, np.squeeze(fake_B))
        print_time_info("Iteration {:0>7} Save the images...".format(counter))
  
    ########################################################
    #                       testing                        #
    ########################################################   
    
    def test(self):
        checker, before_counter = self.load_model()
        if not checker:
            print_time_info("There isn't any ready model, quit.")
            sys.quit()

        for i in range(32):
            batch_A, batch_B = self.get_batch(self.batch_size, is_random=True)
            fake_A, fake_B, cycle_A, cycle_B = self.sess.run(
                    [self.fake_A, self.fake_B, self.cycle_A, self.cycle_B], feed_dict={self.real_A: batch_A, self.real_B: batch_B})
            fake_A, fake_B = fake_A[0], fake_B[0]
            cycle_A, cycle_B = cycle_A[0], cycle_B[0]
            image_real_A_path = os.path.join(self.images_dir, "{}_test{}_real_A.jpg".format(before_counter, i))
            image_real_B_path = os.path.join(self.images_dir, "{}_test{}_real_B.jpg".format(before_counter, i))
            image_fake_A_path = os.path.join(self.images_dir, "{}_test{}_fake_A.jpg".format(before_counter, i))
            image_fake_B_path = os.path.join(self.images_dir, "{}_test{}_fake_B.jpg".format(before_counter, i))
            image_cycle_A_path = os.path.join(self.images_dir, "{}_test{}_cycle_A.jpg".format(before_counter, i))
            image_cycle_B_path = os.path.join(self.images_dir, "{}_test{}_cycle_B.jpg".format(before_counter, i))
            imsave(image_real_A_path, np.squeeze(batch_A))
            imsave(image_real_B_path, np.squeeze(batch_B))
            imsave(image_fake_A_path, np.squeeze(fake_A))
            imsave(image_fake_B_path, np.squeeze(fake_B))
            imsave(image_cycle_A_path, np.squeeze(cycle_A))
            imsave(image_cycle_B_path, np.squeeze(cycle_B))

        print_time_info("Testing end!")
    
    ########################################################
    #                   data processing                    #
    ######################################################## 

    def prepare_data(self):
        if os.path.isdir(os.path.join(self.data_dir, self.dataset)) == False:
            print_time_info("Haven't download the dataset yet...download it now")
            subprocess.run(["bash", self.download_script, self.dataset, self.data_dir])
        self.data_dir = os.path.join(self.data_dir, self.dataset)
        if self.is_train:
            self.images_A = glob.glob(os.path.join(self.data_dir, "trainA/*.jpg"))
            self.images_B = glob.glob(os.path.join(self.data_dir, "trainB/*.jpg"))
        else:
            self.images_A = glob.glob(os.path.join(self.data_dir, "testA/*.jpg"))
            self.images_B = glob.glob(os.path.join(self.data_dir, "testB/*.jpg"))
        print_time_info("Domain A: {} images, Domain B: {} images".format(len(self.images_A), len(self.images_B)))
        
        self.counter_A = self.counter_B = 0
        if self.is_train:
            random.shuffle(self.images_A)
            random.shuffle(self.images_B)

    def get_batch(self, batch_size=1, is_random=False):
        if self.counter_A + batch_size >= len(self.images_A) and not is_random: 
            self.counter_A = 0
            random.shuffle(self.images_A)
        if self.counter_B + batch_size >= len(self.images_B) and not is_random:
            self.counter_B = 0
            random.shuffle(self.images_B)
        if self.channels == 1: grayscale = True
        else: grayscale = False
        
        if is_random:
            images_A_path = np.random.choice(self.images_A, batch_size)
            images_B_path = np.random.choice(self.images_B, batch_size)
        else:
            images_A_path = self.images_A[self.counter_A: self.counter_A + batch_size]
            images_B_path = self.images_B[self.counter_B: self.counter_B + batch_size]
        
        batch_A = get_images(
            images_path = images_A_path,
            input_shape = self.input_shape, 
            output_shape = self.output_shape,
            crop = self.crop, 
            grayscale = grayscale)
        
        batch_B = get_images(
            images_path = images_B_path,
            input_shape = self.input_shape, 
            output_shape = self.output_shape,
            crop = self.crop, 
            grayscale = grayscale)

        if not is_random: 
            self.counter_A += batch_size
            self.counter_B += batch_size
        return batch_A, batch_B

    ########################################################
    #                  load and save model                 #
    ########################################################   
    
    def load_model(self):
        import re
        print_time_info("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print_time_info("Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print_time_info("Failed to find a checkpoint")
            return False, 0

    def save_model(self, counter):
        model_name = os.path.join(self.save_dir, "{}.ckpt".format(counter))
        print_time_info("Saving checkpoint...")
        self.saver.save(self.sess, model_name, global_step=counter)


