from resnet.resnet import resnet
import tensorflow as tf
import tensorflow.contrib.slim as slim
import util.data_pipe
from ultra_lanenet.data_stream import data_stream
import tusimple_process.ultranet_comm
import cv2
from ultra_lanenet.similarity_loss import similaryit_loss
from ultra_lanenet.similarity_loss import structural_loss
from ultra_lanenet.similarity_loss import cls_loss
import logging
import numpy as np
from tusimple_process.create_label import tusimple_label
import os

class ultra_lane():
    def __init__(self):
        #height = 720
        self._row_anchors = tusimple_process.ultranet_comm.ROW_ANCHORS
        self._cells = tusimple_process.ultranet_comm.CELLS
        self._lanes = tusimple_process.ultranet_comm.LANES
        self.cls_label_handle = tusimple_label()
        return

    def make_net(self, x, trainable=True, reuse=False):
        resnet_model = resnet()
        resnet_model.resnet18(x, self._cells+1, trainable, reuse)
        x3 = resnet_model.layer3
        x4 = resnet_model.layer4
        x5 = resnet_model.layer5

        total_dims = (self._cells + 1) * len(self._row_anchors) * self._lanes
        fc = slim.conv2d(x5, 8, [1, 1], 1, padding='SAME', reuse=reuse, scope='fc-1')
        fc = tf.reshape(fc, shape=(-1, 1800))
        fc = tf.contrib.layers.fully_connected(fc, 2048, scope='line1', reuse=reuse, activation_fn=tf.nn.relu)
        fc = tf.contrib.layers.fully_connected(fc, total_dims, scope='line2', reuse=reuse, activation_fn=None)
        group_cls = tf.reshape(fc, shape=(-1, len(self._row_anchors), self._lanes, self._cells+1))

        return group_cls

    def loss(self, group_cls, label):
        cls = cls_loss(group_cls, label, self._cells+1)
        sim = similaryit_loss(group_cls)
        shp = structural_loss(group_cls)

        return cls, sim, shp, tf.argmax(slim.softmax(group_cls), axis=-1)

    def create_train_pipe(self, pipe_handle, config, batch_size, trainable=True, reuse=False, file_name='train_files.txt'):
        train_data_handle = data_stream(config['image_path'], config['img_width'], config['img_height'], file_name)
        src_tensor, label_tensor, cls_tensor, names_tensor = train_data_handle.create_img_tensor()
        #train_data_handle.pre_process_img(src_tensor[0], label_tensor[0], cls_tensor[0], names_tensor[0])
        src_img_queue, label_queue, ground_cls_queue, names_queue = pipe_handle.make_pipe(batch_size, (src_tensor, label_tensor, cls_tensor, names_tensor), train_data_handle.pre_process_img)
        group_cls = self.make_net(src_img_queue, trainable, reuse)
        cls_loss_tensor, sim_loss_tensor, shp_loss_tensor, predict_rows = self.loss(group_cls, ground_cls_queue)
        total_loss_tensor = cls_loss_tensor + sim_loss_tensor + shp_loss_tensor
        return total_loss_tensor, cls_loss_tensor, sim_loss_tensor, shp_loss_tensor, src_img_queue, ground_cls_queue, predict_rows, names_queue

    def train(self, config):
        pipe_handle = util.data_pipe.data_pipe(config['cpu_cores'])
        save_path = config['out_path'] + '/out_img'
        os.makedirs(save_path, exist_ok=True)
        model_path = config['out_path'] + '/model_path/'
        os.makedirs(model_path, exist_ok=True)

        with tf.device(config['device']):
            #train
            total_loss, cls_loss, sim_loss, shp_loss, src_img, ground_cls, predict_cls, img_name = self.create_train_pipe(pipe_handle, config, config['batch_size'])
            total_loss_summary = tf.summary.scalar(name='total-loss', tensor=total_loss)
            cls_loss_summary = tf.summary.scalar(name='cls-loss', tensor=cls_loss)

            b, w, h, c = ground_cls.get_shape().as_list()
            ground_label = tf.cast(tf.reshape(ground_cls, (b, w, h)), dtype=predict_cls.dtype)
            correct_label = tf.cast(tf.equal(ground_label, predict_cls), dtype=tf.float32)
            precision = tf.reduce_sum(correct_label) / (1.0*w*h*b*c)
            precision_summary = tf.summary.scalar(name='precision', tensor=precision)

            global_step = tf.train.create_global_step()
            learning_rate = tf.train.exponential_decay(config['learning_rate'], global_step, config['decay_steps'], config['decay_rate'])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=config['epsilon'])
            train_op = slim.learning.create_train_op(total_loss, optimizer)
            ls_summary = tf.summary.scalar(name='learning-rate', tensor=learning_rate)

            #valid
            valid_total_loss, valid_cls_loss, valid_sim_loss, valid_shp_loss, valid_src_img, valid_ground_cls, valid_predict_cls, valid_img_name = self.create_train_pipe(pipe_handle, config, config['eval_batch_size'], False, True, 'valid_files.txt')
            val_total_loss_summary = tf.summary.scalar(name='val-total-loss', tensor=valid_total_loss)
            val_cls_loss_summary = tf.summary.scalar(name='val-cls-loss', tensor=valid_cls_loss)

            train_summary_op = tf.summary.merge([total_loss_summary, cls_loss_summary, ls_summary, val_total_loss_summary, val_cls_loss_summary, precision_summary])

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                summary_writer = tf.summary.FileWriter(config['out_path'] + '/summary')
                summary_writer.add_graph(sess.graph)

                min_loss = float('inf')
                for step in range(config['train_epoch']):

                    _, total_loss_val, cls_loss_val, sim_loss_val, shp_loss_val, train_summary, gs, lr, src_img_val, ground_cls_val, predict_cls_val, img_name_val, p = sess.run([train_op, total_loss, cls_loss, sim_loss, shp_loss, train_summary_op, global_step, learning_rate, src_img, ground_cls, predict_cls, img_name, precision])

                    total_loss_1 = cls_loss_val + sim_loss_val + shp_loss_val

                    summary_writer.add_summary(train_summary, global_step=gs)

                    logging.info('train model: gs={},  loss={} {}={}+{}+{}, precision={}, lr={}'.format(gs, total_loss_val, total_loss_1, cls_loss_val, sim_loss_val, shp_loss_val, p, lr))

                    if (step + 1) % config['update_mode_freq'] == 0:
                        valid_total_loss_val, valid_src_img_val, valid_ground_cls_val, valid_predict_cls_val, valid_img_name_val = sess.run([valid_total_loss, valid_src_img, valid_ground_cls, valid_predict_cls, valid_img_name])
                        self.match_coordinate(valid_src_img_val.astype(np.uint8), valid_ground_cls_val, valid_predict_cls_val, save_path, step, valid_img_name_val)
                        logging.info('valid model: gs={},  loss={}, lr={}'.format(gs, valid_total_loss_val, lr))
                        print('valid model: gs={},  loss={}, lr={}'.format(gs, valid_total_loss_val, lr))
                        print('train model: gs={},  loss={},[{},{},{}], precision={}, lr={}'.format(gs, total_loss_val, cls_loss_val, sim_loss_val, shp_loss_val, p, lr))
                        if min_loss > total_loss_val:
                            saver.save(sess, model_path, global_step=gs)
                            logging.info('update model loss from {} to {}'.format(min_loss, total_loss_val))

                    min_loss = min(min_loss, total_loss_val)
        return

    def match_coordinate(self, imgs, ground_cls, predict_cls, save_path, epoch, names):
        batch, h, w, c = imgs.shape

        for b in range(batch):
            label_lane = self.cls_label_handle.rescontruct(ground_cls[b][:, :, 0], imgs[b].copy())
            predict_lane = self.cls_label_handle.rescontruct(predict_cls[b], imgs[b].copy())
            all_img = np.hstack([label_lane, predict_lane])
            cv2.imwrite(save_path+'/'+str(epoch)+'-'+str(b)+'-'+str(names[0])+'.png', all_img)
        return

