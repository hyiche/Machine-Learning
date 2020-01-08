from help_file import Model_cGAN as model
from Record2Tensor import *

name = model.name

path_tfrecords_train = "./input/train400_mos2.tfrecords"
path_checkpoint = "./checkpoint/"
path_image_train = "./image_train/"

chech_epoch = 2000
num_epoch = 1000
data_epoch = num_epoch * 2

max_steps = 400
print_epoch = 5
save_img_epoch = 5
save_check_step = 400
