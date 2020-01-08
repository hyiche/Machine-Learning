from utils import *
import numpy as np
import numpy.linalg as nlg
import sparse_coding

Dictionary = np.load('experiment_dictionary_16.npz')
D1, D2 = Dictionary['D1'], Dictionary['D2']
x1, x2 = Dictionary['x1'], Dictionary['x2']
img1_mean, img2_mean = Dictionary['image_1_mean'], Dictionary['image_2_mean']
y1, y2 = Dictionary['y1'], Dictionary['y2']

print(" ||D1^T * D2|| =", "%1.4f" % nlg.norm(np.transpose(D1)@D2),
      " ||D1^T * D1|| =", "%1.4f" % nlg.norm(np.transpose(D1)@D1),
      " ||D2^T * D2|| =", "%1.4f" % nlg.norm(np.transpose(D2)@D2))

image_1 = y1
image_2 = y2

""" set i """
i = 0

""" build A """
residual_y1 = image_1 - Dx2image(D1, x1)
residual_y2 = image_2 - Dx2image(D2, x2)

residual_y1 = image2patches(residual_y1, patch_size=8)
residual_y2 = image2patches(residual_y2, patch_size=8)

residual_y1 = patches2flat(residual_y1)
residual_y2 = patches2flat(residual_y2)

# a = residual_y1[i, :].reshape(1, 64)
# b = residual_y2[i, :].reshape(1, 64)
A = np.concatenate((residual_y1, residual_y2), axis=0)

# """ test """
# D1x1 = D1@x1
# Dx = D1x1[:, i].reshape(64, 1)
# print(A@Dx)
# print(nlg.norm(residual_y1@D1x1))

""" build y mixture"""
y_mixture = y1 + y2
y_mixture = image2DLtrain(y_mixture, patch_size=8)

""" build initial """
# x1_i = x1[:, i].reshape(32, 1)
# x2_i = x2[:, i].reshape(32, 1)

""" sparse coding test """
#sc = sparse_coding.SparseCoding2D1A(D1=D1, D2=D2, A=A,
#                                    lambda_1=0.01, lambda_2=0.01, lambda_3=1, lambda_4=1, lambda_5=0.01, lambda_6=0.01)

sc = sparse_coding.SparseCoding2D(D1=D1, D2=D2)
print("sparse coding (soft)")
x1_sc, x2_sc = sc.get_x(y=y_mixture, x1=None, x2=None, num_iter=3000, proximal_use="soft", soft_coef=0.01)
print("sparse coding (greedy)")
x1_sc, x2_sc = sc.get_x(y=y_mixture, x1=x1_sc, x2=x2_sc, num_iter=3000, proximal_use="greedy", n_nonzero=3)

print('y1 psnr: ', psnr_wo_mean(Dx2image(D1, x1_sc), y1),
      'y2 psnr: ', psnr_wo_mean(Dx2image(D2, x2_sc), y2),
      'D1x1 psnr: ', psnr_wo_mean(Dx2image(D1, x1_sc), Dx2image(D1, x1)),
      'D2x2 psnr: ', psnr_wo_mean(Dx2image(D2, x2_sc), Dx2image(D2, x2)))

plot_Dx(D1, x1_sc)
plot_Dx(D2, x2_sc)

print("done")
