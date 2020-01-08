import numpy as np
import numpy.linalg as nlg
from utils import *
from two_orthogonal_dictionary_learning import TwoOrthogonalDictionaryLearning as TODL

"""Train Dictionary"""

# # load image
# image_1 = load_image(image_path='/home/kumi/Downloads/picture/house.png')
# image_2 = load_image(image_path='/home/kumi/Downloads/picture/pic1.png')
#
# # mean
# image_1_mean = block_mean(image_1, patch_size=8)  # (256, 256)
# image_2_mean = block_mean(image_2, patch_size=8)  # (256, 256)
#
# # remove mean and normalize
# image_1 = normalize_image_wo_mean(image_1 - image_1_mean)  # (256, 256)
# image_2 = normalize_image_wo_mean(image_2 - image_2_mean)  # (256, 256)
# house = image_1
# texture = image_2
#
# # turn into 8 x 8 patches
# image_1 = image2patches(image_1, patch_size=8)  # (1024, 8, 8, 1)
# image_2 = image2patches(image_2, patch_size=8)  # (1024, 8, 8, 1)
#
# # flat the patches
# image_1 = patches2flat(image_1)  # (1024, 64)
# image_2 = patches2flat(image_2)  # (1024, 64)
#
# # transpose image_1 and image_2
# image_1 = np.transpose(image_1)  # (64, 1024)
# image_2 = np.transpose(image_2)  # (64, 1024)


"""load image"""
image = np.load('y1_and_y2_D32_nonzero3.npz')
D1, D2 = image['D1'], image['D2']
x1, x2 = image['x1'], image['x2']
image_1, image_2 = image['image_1'], image['image_2']
house, texture = image['house'], image['texture']
image_1_mean, image_2_mean = image['image_1_mean'], image['image_2_mean']
y1 = Dx2image(D1, x1)
y2 = Dx2image(D2, x2)
plot_image(y1)
plot_image(y2)
print(psnr_wo_mean(y1, house), psnr_wo_mean(y2, texture))
y1_t = image2DLtrain(y1, patch_size=8)
y2_t = image2DLtrain(y2, patch_size=8)

"""init the dictionary and x"""
# np.random.seed(1)
# number_of_atoms = 32
# D1 = dictionary_normalize(np.random.randn(64, number_of_atoms))
# D2 = dictionary_normalize(np.random.randn(64, number_of_atoms))
#
# x1 = np.random.randn(number_of_atoms, 1024)
# x2 = np.random.randn(number_of_atoms, 1024)

"""train D and x with greedy algorithm"""
dictionary_learning = TODL(D1=D1, D2=D2, x1=x1, x2=x2, lmbda=100, lmbda1=0, lmbda2=0, n_nonzero_coefs=3)
dictionary_learning.train(y1=y1_t, y2=y1_t, num_iter=1000, proximal_use='greedy')
D1, D2 = dictionary_learning.get_dictionary()
x1, x2 = dictionary_learning.get_x()

repeat = 30
L_thres_x1 = np.zeros(shape=(1024, 1))
L_thres_x2 = np.zeros(shape=(1024, 1))

for step in range(repeat):
    print(step+1, '/', repeat)
    """train x, fix D with soft threshold"""
    dictionary_learning = TODL(D1=D1, D2=D2, x1=x1, x2=x2, lmbda=100, lmbda1=0, lmbda2=0, n_nonzero_coefs=3)
    dictionary_learning.train_fix_D(y1=y1_t, y2=y2_t, num_iter=1000, proximal_use='soft_mean', soft_coef=0.01)
    x1, x2 = dictionary_learning.get_x()

    """update lambda"""
    L_thres_x1 = np.mean(abs(x1), axis=0)
    L_thres_x2 = np.mean(abs(x2), axis=0)

    """train D and x with hard threshold"""
    dictionary_learning = TODL(D1=D1, D2=D2, x1=x1, x2=x2, lmbda=100, lmbda1=0, lmbda2=0, n_nonzero_coefs=3)
    dictionary_learning.train(y1=y1_t, y2=y2_t, num_iter=1000, proximal_use='greedy',
                              l_thres_x1=L_thres_x1, l_thres_x2=L_thres_x2)
    D1, D2 = dictionary_learning.get_dictionary()
    x1, x2 = dictionary_learning.get_x()

"""save data"""
np.savez('experiment_dictionary_17.npz',
         D1=D1, D2=D2, x1=x1, x2=x2, house=house, texture=texture, y1=y1, y2=y2,
         image_1_mean=image_1_mean, image_2_mean=image_2_mean)
print("save successfully")

# plot
plot_Dx(D1, x1)
plot_Dx(D2, x2)
