__version__ = '0.0.2'

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import torch as ch
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.allocation_query import AllocationQuery
from dataclasses import replace

class ToTFImage(Operation):
    """Go from Pyotrch to TensorFlow format for images (B x H x W x C).
    """
    def __init__(self):
        super().__init__()

    def generate_code(self):
        def to_tf_image(inp: ch.Tensor, dst):
            inp = inp.permute([0, 2, 3, 1])
            if not inp.is_contiguous():
                dst[:inp.shape[0]] = inp.contiguous()
                return dst[:inp.shape[0]]
            return inp

        return to_tf_image

    def declare_state_and_memory(self, previous_state):
        alloc = None
        C, H, W = previous_state.shape
        new_type = previous_state.dtype

        alloc = AllocationQuery((H, W, C), dtype=new_type)
        return replace(previous_state, shape=(H, W, C), dtype=new_type), alloc


class FFCVKerasSequence(tf.keras.utils.Sequence):
    """Wraps an FFCV Loader into a Keras Sequence for use with tensorflow.
    """
    def __init__(self, dl):
        self.dl = dl
        self.iter = None

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        if self.iter is None:
            self.iter = iter(self.dl)
        try:
            content = next(self.iter)
            result = []
            for tensor in content:
                if ch.is_tensor(tensor):
                    if tensor.is_cuda:
                        tensor = ch.utils.dlpack.to_dlpack(tensor)
                        tensor = tf.experimental.dlpack.from_dlpack(tensor)
                    else:
                        tensor = tensor.numpy()
                result.append(tensor)
            return tuple(result)

        except StopIteration:
            self.iter = None
            return self.__getitem__(index)

    def __len__(self):
        return len(self.dl)
