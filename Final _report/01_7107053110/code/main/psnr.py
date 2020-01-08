import tensorflow as tf



def read_img(path):
    return tf.image.decode_image(tf.read_file(path))


def psnr(tf_img1, tf_img2):
    return tf.image.psnr(tf_img1, tf_img2, max_val=255)

def _main():
    total = 0
    num = 5
    for i in range(1, num+1):
        target = read_img('D:/exp_6AE/1/400/addchangloss/500/test_unseen/{}target.png'.format(i))
        output = read_img('D:/exp_6AE/1/400/addchangloss/500/test_unseen/{}output.png'.format(i))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            y1 = sess.run(psnr(target, output))
            total += y1
            print("order:", i)
            print("psnr:", y1)
            print(" ")
    print('avg:', total/num)

if __name__ == '__main__':
    _main()