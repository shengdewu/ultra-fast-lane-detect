import tensorflow as tf
import ultra_lanenet.similarity_loss

if __name__ == '__main__':
    logit = tf.constant([[[[1, 0, 2], [2, 1, 0]], [[3, 1, 2], [4, 3, 1]]]], dtype=tf.float32)
    label = tf.constant([[[[1], [0]], [[0], [1]]]], dtype=tf.uint8)
    x = ultra_lanenet.similarity_loss.cls_loss(logit, label, 3)
    x1 = ultra_lanenet.similarity_loss.similaryit_loss(logit)
    #x2 = ultra_lanenet.similarity_loss.structural_loss(logit)
    with tf.Session() as sess:
        v, v1 = sess.run([x, x1])
        print(v)
        print(v1)
        #print(v2)