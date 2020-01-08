import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_normalize_image(image_path):
    # load normalized image data
    return normalize_image(np.array(load_image(image_path)))


def load_image(image_path):
    # load gray image
    return np.array(Image.open(image_path).convert('L'))


def plot_image(image):
    # show gray image
    plt.imshow(image, cmap='gray')
    plt.show()
    return None


def normalize_image_std(image):
    # do zero-mean normalization on image
    return (image - np.mean(image))/np.std(image)


def normalize_image(image):
    return (np.array(image) / 127.5) - 1


def normalize_image_wo_mean(image):
    return np.array(image) / 255


def image2patches(image, patch_size):
    # e.g. image (256, 256), patch_size=16 -> 256 16x16x1 patches (256, 16, 16, 1)
    h, w = image.shape
    image_patches = []
    for i in range(int(w/patch_size)):
        for j in range(int(h/patch_size)):
            image_patches.append(image[patch_size*i: patch_size*i+patch_size,
                                 patch_size*j: patch_size*j+patch_size].reshape(patch_size, patch_size, 1))
    return np.array(image_patches)


def patches2flat(image):
    quantity, h, w, c = image.shape
    image_flat = []
    for i in range(quantity):
        image_flat.append(image[i].reshape(h*w))
    return np.array(image_flat)


def flat2patches(image):
    quantity, flat_size = image.shape
    patches_size = int(math.sqrt(flat_size))
    image_patches = []
    for i in range(quantity):
        image_patches.append(image[i].reshape(patches_size, patches_size, 1))
    return np.array(image_patches)


def patches2image(image):
    # (1024, 8, 8, 1) -> (256, 256)
    quantity, h, w, c = image.shape
    size = int(int(math.sqrt(quantity))*h)
    reconstructed_image = np.zeros(shape=(size, size))
    num_patch = int(math.sqrt(quantity))
    for i in range(num_patch):
        for j in range(num_patch):
            reconstructed_image[h*i:h*i+h, w*j:w*j+w] = np.reshape(image[num_patch*i+j], (h, w))
    return reconstructed_image


def flat2image(image):
    return patches2image(flat2patches(image))


def extract_flat_patches(image, patch_size):
    # 128*128 -> 256*8*8 -> 256*64
    h, w = image.shape
    img_cut = []
    for i in range(int(w/patch_size)):
        for j in range(int(h/patch_size)):
            img_cut.append(image[patch_size*i: patch_size*i+patch_size,
                                 patch_size*j: patch_size*j+patch_size].reshape(patch_size*patch_size).tolist())
    return np.array(img_cut)


def dictionary_normalize(dictionary):
    # normalize each column of dictionary
    for i in range(dictionary.shape[1]):
        dictionary[:, i] = dictionary[:, i] / np.linalg.norm(dictionary[:, i])
    return dictionary


def l0_proximal_x(x, n_nonzero_coefs):
    for c in range(x.shape[1]):
        y = np.argsort(np.abs(x[:, c]))[::-1]
        x_new = np.zeros(x.shape[0])
        for i in y[0:n_nonzero_coefs]:
            x_new[i] = x[i, c]
        x[:, c] = x_new
    return x


def plot_loss(loss, name):
    y = loss
    x = np.arange(0, len(loss))
    plt.figure()
    plt.plot(x, y)
    plt.ylabel("loss")
    plt.xlabel("iters")
    plt.title(name)

    plt.show()


def psnr(img1, img2):
    mse = np.mean(((img1+1)/2. - (img2+1)/2.) ** 2)
    if mse < 1.0e-10:
        return 100
    pixel_max = 1
    return 20 * math.log10(pixel_max/math.sqrt(mse))


def psnr_wo_mean(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def block_mean(image, patch_size):
    image = patches2flat(image2patches(image=image, patch_size=patch_size))
    return flat2image(np.multiply(np.mean(image.T, axis=0), np.ones_like(image.T)).T)


def get_lambda_threshold(x, non_zero):
    return np.sort(a=abs(x), axis=0)[x.shape[0] - non_zero]


def plot_Dx(D, x):
    Dx = (D @ x)
    Dx = flat2image(Dx.T)
    plot_image(Dx)


def Dx2image(D, x):
    return flat2image((D @ x).T)


def image2DLtrain(image, patch_size):
    return np.transpose(patches2flat(image2patches(image, patch_size=patch_size)))
