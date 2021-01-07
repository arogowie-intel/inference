# coding=utf-8
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import array
import os
import sys

import mlperf_loadgen as lg
import numpy as np
from squad_QSL import get_squad_QSL
import paddle
import paddle.fluid as fluid


class BERT_PaddleOneDNN_SUT():
    def __init__(self, args):

        print("Loading ONNX model...")
        model_dir = "build/data/pdpd/inference_model"
        model_path = os.path.join(model_dir, "__model__")
        model_params = os.path.join(model_dir, "__params__")
        config = None
        if os.path.exists(model_params):
            config = fluid.core.AnalysisConfig(model_path, model_params)
        else:
            config = fluid.core.AnalysisConfig(model_dir)
        config.disable_gpu()
        config.enable_mkldnn()
        nthr = int(os.environ.get('OMP_NUM_THREADS', 40))
        config.set_cpu_math_library_num_threads(nthr)
        # config.disable_glog_info()
        config.enable_memory_optim()

        # Those two passess brake-up things around matmul op with output: `matmul_3.tmp_0`
        config.delete_pass("reshape_transpose_matmul_mkldnn_fuse_pass")
        config.delete_pass("matmul_transpose_reshape_fuse_pass")
        config.switch_ir_optim(True)
        config.switch_use_feed_fetch_ops(False)
        # config.switch_ir_debug(True)

        # self.predictor = fluid.core.create_predictor(config)  # v1
        self.predictor = fluid.core.create_paddle_predictor(config)

        # set-up input/output tensors
        self.inputs = self.predictor.get_input_names()
        input_kv = [(i, self.predictor.get_input_tensor(i)) for i in self.inputs]
        self.input_tensors = dict(input_kv)
        # self.input_tensors = self.predictor.get_input_handle(self.self.inputs[0]) # v1

        self.outputs = self.predictor.get_output_names()
        output_kv = [(o, self.predictor.get_output_tensor(o)) for o in self.outputs]
        self.output_tensors = dict(output_kv)
        # output_tensors = self.predictor.get_output_handle(self.self.outputs[0]) # v1

        print("Constructing SUT...")
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries, self.process_latencies)
        print("Finished constructing SUT.")

        total_samples = None
        perf_samples = None
        if args.count:
            total_samples = args.count
            perf_samples = min(args.count, 500)
        self.qsl = get_squad_QSL(total_samples, perf_samples)

    def get_feature_desc(self, feature):
        return {
            "x2paddle_input_ids": np.array(feature.input_ids).astype(np.int64)[np.newaxis, :],
            "x2paddle_input_mask": np.array(feature.input_mask).astype(np.int64)[np.newaxis, :],
            "x2paddle_segment_ids": np.array(feature.segment_ids).astype(np.int64)[np.newaxis, :]
        }

    def warmup(self):
        for i in range(10):
            feature = self.qsl.get_features(i)
            fd = self.get_feature_desc(feature)
            # import ipdb; ipdb.set_trace()
            for k, v in fd.items():
                self.input_tensors[k].copy_from_cpu(v)
            self.predictor.zero_copy_run()

    def issue_queries(self, query_samples):
        for i in range(len(query_samples)):
            # import ipdb; ipdb.set_trace()
            eval_features = self.qsl.get_features(query_samples[i].index)
            fd = self.get_feature_desc(eval_features)
            for k, v in fd.items():
                self.input_tensors[k].copy_from_cpu(v)

            self.predictor.zero_copy_run()
            # self.predictor.run() # v1

            scores = list()
            for k, v in self.output_tensors.items():
                scores.append(v.copy_to_cpu())

            output = np.stack(scores, axis=-1)[0]

            response_array = array.array("B", output.tobytes())
            bi = response_array.buffer_info()
            response = lg.QuerySampleResponse(query_samples[i].id, bi[0], bi[1])
            lg.QuerySamplesComplete([response])

    def flush_queries(self):
        pass

    def process_latencies(self, latencies_ns):
        pass

    def __del__(self):
        print("Finished destroying SUT.")


def get_paddle_sut(args):
    return BERT_PaddleOneDNN_SUT(args)
