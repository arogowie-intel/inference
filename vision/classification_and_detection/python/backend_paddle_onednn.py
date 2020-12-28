"""
PaddlePaddle backend
"""
# pylint: disable=missing-docstring
import os
import numpy as np

import paddle
import paddle.fluid as fluid
import backend


class BackendPaddleOneDNN(backend.Backend):
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.inputs = None
        self.outputs = None
        self.input_tensors = None
        self.output_tensors = None

    def version(self):
        return paddle.__version__

    def name(self):
        return "paddlepaddle"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        model_dir, _, _ = model_path.rpartition(os.path.sep)
        params_path = os.path.join(model_dir, "__params__")
        config = None
        if os.path.exists(params_path):
            config = fluid.core.AnalysisConfig(model_path, params_path)
        else:
            config = fluid.core.AnalysisConfig(model_dir)
        config.disable_gpu()
        config.enable_mkldnn()
        nthr = int(os.environ.get('OMP_NUM_THREADS', 40))
        config.set_cpu_math_library_num_threads(nthr)
        config.disable_glog_info()
        config.enable_memory_optim()
        config.switch_ir_optim(True)
        config.switch_use_feed_fetch_ops(False)

        # self.predictor = fluid.core.create_predictor(config)  # v1
        self.predictor = fluid.core.create_paddle_predictor(config)

        self.inputs = inputs
        if self.inputs is None:
            self.inputs = self.predictor.get_input_names()
        input_kv = [(i, self.predictor.get_input_tensor(i)) for i in self.inputs]
        self.input_tensors = dict(input_kv)
        # self.input_tensors = self.predictor.get_input_handle(self.self.inputs[0]) # v1

        self.outputs = outputs
        if self.outputs is None:
            self.outputs = self.predictor.get_output_names()
        output_kv = [(o, self.predictor.get_output_tensor(o)) for o in self.outputs]
        self.output_tensors = dict(output_kv)
        # output_tensors = self.predictor.get_output_handle(self.self.outputs[0]) # v1

        return self

    def predict(self, feed):
        # key = self.inputs[0]
        for k, v in feed.items():
            self.input_tensors[k].copy_from_cpu(v)

        self.predictor.zero_copy_run()
        # self.predictor.run() # v1

        results = list()
        for k, v in self.output_tensors.items():
            results.append(v.copy_to_cpu())
        return results
