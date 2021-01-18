import tensorflow as tf
import numpy as np
from tensorflow_core.python import tf_export
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

class CosineAnnealing(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup, eta_min=0.0, eta_max=1.0, name=None):
        super(CosineAnnealing, self).__init__()
        self.__eta_min = eta_min
        self.__eta_max = eta_max
        self.__warmup = warmup
        self.__name = name
        self.__pi = 3.141592653589793
        return

    def __call__(self, step):
        with ops.name_scope_v2(self.__name or "CosineAnnealing") as name:
            pi = ops.convert_to_tensor(self.__pi, name="pi")

            dtype = pi.dtype
            eta_min = math_ops.cast(self.__eta_min, dtype)
            eta_max = math_ops.cast(self.__eta_max, dtype)
            warmup = math_ops.cast(self.__warmup, dtype)

            global_step_recomp = math_ops.cast(step, dtype)

            rate = ops.convert_to_tensor(0.5, name='initial_rate', dtype=dtype)

            lr = math_ops.multiply(math_ops.multiply(rate, eta_max - eta_min),
                              math_ops.cos(
                                  math_ops.multiply(pi,
                                                    math_ops.div(global_step_recomp, warmup))) + 1, name=name)
        return lr +eta_min

    def get_config(self):
        return {
            'eta_min': self.__eta_min,
            'eta_max': self.__eta_max,
            'warmup': self.__warmup,
            'name': self.__name
        }


@tf_export(v1=["util.CosineAnnealing.cosine_decay"])
def cosine_decay(global_step, warmup, eta_min, eta_max, name=None):

    decayed_lr = CosineAnnealing(warmup, eta_min, eta_max, name)

    decayed_lr = decayed_lr(global_step)

    return decayed_lr