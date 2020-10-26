
import numpy as np
import tensorflow as tf



def eps_greedy_sample(logits, eps):
    '''
    rand_num = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32)
    sample = tf.cond(rand_num < eps,
                     lambda: tf.random.uniform([tf.shape(logits)[0], 1], minval=0, maxval=tf.shape(logits)[1], dtype=tf.int32),
                     lambda: tf.multinomial(logits, 1, output_dtype=tf.int32))
    '''

    rand_num = tf.random_uniform([tf.shape(logits)[0], 1], minval=0, maxval=1, dtype=tf.float32)
    mask = tf.cast(rand_num < eps, tf.int32)

    rand_sample = tf.random_uniform([tf.shape(logits)[0], 1], minval=0, maxval=tf.shape(logits)[1], dtype=tf.int32)
    greedy_sample = tf.multinomial(logits, 1, output_dtype=tf.int32)


    return mask * rand_sample + (1 - mask) * greedy_sample


def gaussian_sample(mean, logvar):
    raise NotImplementedError



if __name__ == '__main__':
    logits = tf.log(tf.constant([[1.], [1.]]))
    eps = tf.placeholder(tf.float32, ())

    with tf.Session() as sess:
        sample = sess.run(eps_greedy_sample(logits, eps), feed_dict={eps: 0.})
        print(sample)
