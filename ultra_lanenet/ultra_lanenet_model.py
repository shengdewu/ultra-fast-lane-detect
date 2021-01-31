from resnet.resnet import resnet
import tensorflow as tf
import tensorflow.contrib.slim as slim
import util.data_pipe
from ultra_lanenet.data_stream import data_stream
import tusimple_process.ultranet_comm
import cv2
import ultra_lanenet.similarity_loss
import logging
import numpy as np
from tusimple_process.create_label import tusimple_label
import os
import util.CosineAnnealing
import math


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
        resnet_model.resnet18(x, trainable, reuse)
        x3 = resnet_model.layer3
        x4 = resnet_model.layer4
        x5 = resnet_model.layer5

        total_dims = (self._cells + 1) * len(self._row_anchors) * self._lanes
        fc = slim.conv2d(x5, 8, [1, 1], 1, padding='SAME', reuse=reuse, scope='fc-1')
        fc = tf.reshape(fc, shape=(-1, 1800))
        fc = tf.contrib.layers.fully_connected(fc, 2048, scope='line1', reuse=reuse, activation_fn=None, weights_initializer=tf.initializers.random_normal(0, 1.0), biases_initializer=tf.initializers.random_normal(0, 1.0))
        fc = tf.nn.relu(fc)
        fc = tf.contrib.layers.fully_connected(fc, total_dims, scope='line2', reuse=reuse, activation_fn=None, weights_initializer=tf.initializers.random_normal(0, 1.0), biases_initializer=tf.initializers.random_normal(0, 1.0))

        group_cls = tf.reshape(fc, shape=(-1, len(self._row_anchors), self._lanes, self._cells+1))

        return group_cls

    def loss(self, group_cls, label):
        cls = ultra_lanenet.similarity_loss.cls_loss(group_cls, label, self._cells+1)
        sim = ultra_lanenet.similarity_loss.similaryit_loss(group_cls)
        shp = ultra_lanenet.similarity_loss.structural_loss(group_cls)

        return cls, sim, shp, tf.argmax(group_cls, axis=-1)

    def create_train_pipe(self, pipe_handle, config, batch_size, trainable=True, reuse=False, file_name='train_files.txt'):
        train_data_handle = data_stream(config['image_path'], config['img_width'], config['img_height'], file_name, config['lanes'])
        src_tensor, label_tensor, cls_tensor, total_img = train_data_handle.create_img_tensor()
        #train_data_handle.pre_process_img(src_tensor[0], label_tensor[0], cls_tensor[0])
        src_img_train_queue, label_queue, ground_cls_queue, src_img_queue = pipe_handle.make_pipe(batch_size, (src_tensor, label_tensor, cls_tensor), train_data_handle.pre_process_img)
        group_cls = self.make_net(src_img_train_queue, trainable, reuse)
        cls_loss_tensor, sim_loss_tensor, shp_loss_tensor, predict_rows = self.loss(group_cls, ground_cls_queue)
        total_loss_tensor = cls_loss_tensor + sim_loss_tensor * config['sim_loss_w'] + shp_loss_tensor * config['shp_loss_w']
        tensor = dict()
        tensor['total_loss'] = total_loss_tensor
        tensor['cls_loss'] = cls_loss_tensor
        tensor['sim_loss'] = sim_loss_tensor
        tensor['shp_loss'] = shp_loss_tensor
        tensor['src_img'] = src_img_queue
        tensor['ground_cls'] = ground_cls_queue
        tensor['predict'] = predict_rows
        tensor['label_img'] = label_queue
        tensor['img_epoch'] = math.ceil(total_img / batch_size)
        tensor['total_epoch'] = config['train_epoch'] * tensor['img_epoch']
        return tensor

    def train(self, config):
        pipe_handle = util.data_pipe.data_pipe(config['cpu_cores'])
        save_path = config['out_path'] + '/out_img'
        os.makedirs(save_path, exist_ok=True)
        model_path = config['out_path'] + '/model_path/'
        os.makedirs(model_path, exist_ok=True)

        with tf.device(config['device']):
            #train
            pipe = self.create_train_pipe(pipe_handle, config, config['batch_size'])
            total_loss_summary = tf.summary.scalar(name='total-loss', tensor=pipe['total_loss'])
            cls_loss_summary = tf.summary.scalar(name='cls-loss', tensor=pipe['cls_loss'])

            b, w, h, c = pipe['ground_cls'].get_shape().as_list()
            ground_label = tf.cast(tf.reshape(pipe['ground_cls'], (b, w, h)), dtype=pipe['predict'].dtype)
            correct_label = tf.cast(tf.equal(ground_label, pipe['predict']), dtype=tf.float32)
            precision = tf.reduce_sum(correct_label) / (1.0*w*h*b*c)
            precision_summary = tf.summary.scalar(name='precision', tensor=precision)

            global_step = tf.train.create_global_step()
            #learning_rate = tf.train.exponential_decay(config['learning_rate'], global_step, config['decay_steps'], config['decay_rate'])
            learning_rate = util.CosineAnnealing.cosine_decay(global_step, config['warmup_iter'], pipe['total_epoch'], config['end_learning_rate'], config['learning_rate'])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = slim.learning.create_train_op(pipe['total_loss'], optimizer)
            ls_summary = tf.summary.scalar(name='learning-rate', tensor=learning_rate)

            #valid
            valid_pipe = self.create_train_pipe(pipe_handle, config, config['eval_batch_size'], False, True, 'valid_files.txt')
            val_total_loss_summary = tf.summary.scalar(name='val-total-loss', tensor=valid_pipe['total_loss'])
            val_cls_loss_summary = tf.summary.scalar(name='val-cls-loss', tensor=valid_pipe['cls_loss'])

            train_summary_op = tf.summary.merge([total_loss_summary, cls_loss_summary, ls_summary, val_total_loss_summary, val_cls_loss_summary, precision_summary])

            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                summary_writer = tf.summary.FileWriter(config['out_path'] + '/summary')
                summary_writer.add_graph(sess.graph)

                epoch = -1
                for step in range(pipe['total_epoch']):
                    if step % pipe['img_epoch'] == 0:
                        epoch += 1
                    _, total_loss, p, gs, lr, train_summary = sess.run([train_op, pipe['total_loss'], precision, global_step, learning_rate, train_summary_op])

                    #summary_writer.add_summary(train_summary, global_step=gs)

                    valid_total_loss, valid_src_img, val_label_img, valid_ground_cls, valid_predict = sess.run([valid_pipe['total_loss'], valid_pipe['src_img'], valid_pipe['label_img'], valid_pipe['ground_cls'], valid_pipe['predict']])
                    self.match_coordinate(valid_src_img.astype(np.uint8), val_label_img, valid_ground_cls, valid_predict, save_path, epoch)
                    logging.info('train model: gs={},  loss={}, precision={}, lr={}, valid_loss={}'.format(gs, total_loss, p, lr, valid_total_loss))

                    if step > config['update_mode_freq'] and step % config['update_mode_freq'] == 0:
                        saver.save(sess, model_path, global_step=gs)
                        print('update model loss from {} :{}'.format(step, total_loss))

        return

    def match_coordinate(self, src_img, label_img, ground_cls, predict_cls, save_path, epoch):
        batch, h, w, c = src_img.shape

        for b in range(batch):
            #label = np.repeat(label_img[b] * 60, 3, axis=2)
            label_lane = self.rescontruct(ground_cls[b][:, :, 0])
            predict_lane = self.rescontruct(predict_cls[b])
            all_img = np.hstack([label_lane, predict_lane])
            cv2.imwrite(save_path+'/'+str(epoch)+'-'+str(b)+'.png', all_img)
        return

    def rescontruct(self, cls_label, show=False):
        color = [(255, 0, 0), (0,255,0), (0, 0, 255), (255, 255, 0)]
        lane_img = np.zeros((720, 1280, 3), dtype=np.uint8)

        for i in range(cls_label.shape[1]):
            pti = cls_label[:, i]
            to_pts = [int(pt * tusimple_process.ultranet_comm.ORIGINAL_W / tusimple_process.ultranet_comm.CELLS) if pt != self._cells else -2 for pt in pti]
            points = [(w, h) for h, w in zip(tusimple_process.ultranet_comm.ROW_ANCHORS, to_pts)]
            for l in points:
                if l[0] == -2:
                    continue
                cv2.circle(lane_img, l, radius=3, color=color[i], thickness=3)
        if show:
            cv2.imshow('img', lane_img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        return lane_img

