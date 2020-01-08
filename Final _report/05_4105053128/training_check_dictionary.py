import numpy as np
import numpy.linalg as nlg
from utils import *
from dictionary_learning import DictionaryLearning

# Dictionary = np.load('experiment_dictionary_14.npz')
# D1, D2 = Dictionary['D1'], Dictionary['D2']
# x1, x2 = Dictionary['x1'], Dictionary['x2']
# img1_mean, img2_mean = Dictionary['image_1_mean'], Dictionary['image_2_mean']
#
#
# print(" ||D1^T * D2|| =", "%1.4f" % nlg.norm(np.transpose(D1)@D2),
#       " ||D1^T * D1|| =", "%1.4f" % nlg.norm(np.transpose(D1)@D1),
#       " ||D2^T * D2|| =", "%1.4f" % nlg.norm(np.transpose(D2)@D2))
#
# D1x1 = Dx2image(D1, x1)
# D2x2 = Dx2image(D2, x2)
#
# # load image
# image_1 = load_image(image_path='/home/kumi/Downloads/picture/house.png')
# image_2 = load_image(image_path='/home/kumi/Downloads/picture/pic1.png')
#
# image_1 = normalize_image_wo_mean(image_1 - img1_mean)  # (256, 256)
# image_2 = normalize_image_wo_mean(image_2 - img2_mean)  # (256, 256)
# house = image_1
# texture = image_2
#
# print(psnr_wo_mean(house, D1x1))
# print(psnr_wo_mean(texture, D2x2))
# sign_x1 = np.sign(x1)
# plot_image(D1x1)
# plot_image(D2x2)
#
# print(" ")

# load image
image_1 = load_image(image_path='/home/kumi/Downloads/picture/house.png')
image_2 = load_image(image_path='/home/kumi/Downloads/picture/pic1.png')

# mean
image_1_mean = block_mean(image_1, patch_size=8)  # (256, 256)
image_2_mean = block_mean(image_2, patch_size=8)  # (256, 256)

# remove mean and normalize
image_1 = normalize_image_wo_mean(image_1 - image_1_mean)  # (256, 256)
image_2 = normalize_image_wo_mean(image_2 - image_2_mean)  # (256, 256)
house = image_1
texture = image_2

# turn into 8 x 8 patches
image_1 = image2patches(image_1, patch_size=8)  # (1024, 8, 8, 1)
image_2 = image2patches(image_2, patch_size=8)  # (1024, 8, 8, 1)

# flat the patches
image_1 = patches2flat(image_1)  # (1024, 64)
image_2 = patches2flat(image_2)  # (1024, 64)

# transpose image_1 and image_2
image_1 = np.transpose(image_1)  # (64, 1024)
image_2 = np.transpose(image_2)  # (64, 1024)

# init the dictionary and x
np.random.seed(1)
number_of_atoms = 32
D1 = dictionary_normalize(np.random.randn(64, number_of_atoms))
D2 = dictionary_normalize(np.random.randn(64, number_of_atoms))

x1 = np.random.randn(number_of_atoms, 1024)
x2 = np.random.randn(number_of_atoms, 1024)

# y1 = D1*x1
DL1 = DictionaryLearning(D=D1, x=x1)
DL1.train(y=image_1, num_iter=10000, learning_rate=2e-3, non_zero=3)
D1, x1 = DL1.get_D_and_x()

plot_Dx(D1, x1)
print(psnr_wo_mean(Dx2image(D1, x1), house))

# y2 = D2*x2
DL2 = DictionaryLearning(D=D2, x=x2)
DL2.train(y=image_2, num_iter=10000, learning_rate=2e-3, non_zero=3)
D2, x2 = DL2.get_D_and_x()

plot_Dx(D2, x2)
print(psnr_wo_mean(Dx2image(D2, x2), texture))

"""save data"""
np.savez('y1_and_y2_D32_nonzero3.npz',
         D1=D1, D2=D2, x1=x1, x2=x2, image_1=image_1, image_2=image_2, house=house, texture=texture,
         image_1_mean=image_1_mean, image_2_mean=image_2_mean)
print("save successfully")
