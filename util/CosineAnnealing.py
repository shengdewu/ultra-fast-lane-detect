import tensorflow as tf
import numpy as np
from tensorflow_core.python import tf_export
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

class CosineAnnealing(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_iter, max_iter, eta_min=0.0, eta_max=1.0, name=None, warn_up='linear'):
        super(CosineAnnealing, self).__init__()
        self.__eta_min = eta_min
        self.__eta_max = eta_max
        self.__warmup_iter = warmup_iter
        self.__warmup = warn_up
        self.__max_iter = max_iter
        self.__name = name
        self.__pi = 3.141592653589793
        return

    def __call__(self, step):
        with ops.name_scope_v2(self.__name or "CosineAnnealing") as name:
            pi = ops.convert_to_tensor(self.__pi, name="pi")

            dtype = pi.dtype
            eta_min = math_ops.cast(self.__eta_min, dtype)
            eta_max = math_ops.cast(self.__eta_max, dtype)
            warmup_iter = math_ops.cast(self.__warmup_iter, dtype)
            max_iter = math_ops.cast(self.__max_iter, dtype)
            global_step = math_ops.cast(step, dtype)

            def multi_policy():
                return eta_min + math_ops.div(math_ops.mul(eta_max-eta_min, 1 + math_ops.cos(math_ops.div(math_ops.mul(pi, global_step), max_iter))), 2)

            def linear():
                return math_ops.mul(eta_max, math_ops.div(global_step, warmup_iter))

            if self.__warmup == 'linear':
                return tf.cond(global_step < warmup_iter, linear, multi_policy)
        return multi_policy()

    def get_config(self):
        return {
            'eta_min': self.__eta_min,
            'eta_max': self.__eta_max,
            'warn_up': self.__warn_up,
            'warmup_iter': self.__warmup_iter,
            'max_iter': self.__max_iter,
            'name': self.__name
        }


@tf_export(v1=["util.CosineAnnealing.cosine_decay"])
def cosine_decay(global_step, warmup_iter, max_iter, eta_min, eta_max, name=None, warn_up='linear'):

    decayed_lr = CosineAnnealing(warmup_iter, max_iter, eta_min, eta_max, name, warn_up)

    decayed_lr = decayed_lr(global_step)

    return decayed_lr