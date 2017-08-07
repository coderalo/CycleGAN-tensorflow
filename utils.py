import time
import datetime
import math
import os
import sys

def print_time_info(string):
    T = time.gmtime()
    Y, M, D = T.tm_year, T.tm_mon, T.tm_mday
    h, m, s = T.tm_hour, T.tm_min, T.tm_sec
    print("[{}-{:0>2}-{:0>2} {:0>2}:{:0>2}:{:0>2}] {}".format(Y, M, D, h, m, s, string))

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def valid_input_and_output(input_image_size, output_image_size):
    def valid_size(size):
        if type(size) == list or type(size) == tuple:
            assert len(size) <= 2, "ImageShapeError"
            if len(size) == 2:
                height, width = size
            else:
                height = width = size[0]
        elif type(size) == int:
            height = width = size
        else: assert False, "ImageShapeTypeError"
        return height, width
    
    input_height, input_width = valid_size(input_image_size)
    if output_image_size == None:
        output_height, output_width = input_height, input_width
    else:
        output_height, output_width = valid_size(output_image_size)

    return input_height, input_width, output_height, output_width
     
def check_dir(checkpoint_dir):
    checkpoint = checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print_time_info("{} doesn't exist, create new directory.".format(checkpoint_dir))
    elif not os.path.isdir(checkpoint_dir):
        if not os.path.exists("./model"): os.makedirs("./model")
        print_time_info("{} conflicts, use directory {}".format("./model"))
        checkpoint = "./model"
    else:
        print_time_info("Use the directory {}.".format(checkpoint_dir))

    return checkpoint

def check_log(log_path, training=True):
    if os.path.exists(log_path):
        print_time_info("{} has existed!".format(log_path))
        ans = input("Are you going to truncate the file? (y/[N]) ")
        if not ans == 'y': 
            print("Exit...")
            sys.exit()
        else:
            print("Truncate the file...")
    else:
        print_time_info("{} hasn't existed, create the file...".format(log_path))
    with open(log_path, 'w') as file:
        file.write("counter,errD,errG\n")
    print_time_info("Log file {} is ready!".format(log_path))
    return log_path

