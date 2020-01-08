from help_file import Model_cGAN as model
from Record2Tensor import *
name = model.name

path_tfrecords_test = "./test_input/test_unseen_mos2.tfrecords"
path_checkpoint = "./checkpoint/"
path_image_train = "./test_unseen/"

test_epoch = 1
num_epoch = 2000
data_epoch = num_epoch * 2

max_steps = 5
print_epoch = 100
save_img_epoch = 200
save_check_epoch = 100


