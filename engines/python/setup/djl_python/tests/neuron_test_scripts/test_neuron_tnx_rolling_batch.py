#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import unittest
import warnings
import json
import os
import sys
from collections import defaultdict

try:
    from djl_python.transformers_neuronx import TransformersNeuronXService
    SKIP_TEST = False
except ImportError:
    SKIP_TEST = True

expected_text_30 = {
    "TinyLlama/TinyLlama-1.1B-Chat-v0.6": {
        1:
        "Hello, my name is [Your Name] and I am a [Your Job Title] at [Your Company Name]. I am interested in learning more about your company'",
        2:
        'The president of the United States is a man named Donald Trump.\n\n2. The president of the United States is a man named Donald Trump.\n\n3. The president',
        3:
        'The capital of France is Paris.\n\n2. The capital of the United States is Washington, D.C.\n\n3. The capital of Canada is Ott',
        4:
        "The future of AI is bright, and it's not just in the realm of science fiction. Artificial intelligence is already being used in a wide range of industries",
    }
}

OPTIMUM_CAUSALLM_MODEL_TYPES = {"gpt2", "opt", "bloom", "llama", "mistral"}
OPTIMUM_CAUSALLM_CONTINUOUS_BATCHING_MODELS = {"llama", "mistral"}
VLLM_CONTINUOUS_BATCHING_MODELS = {"llama"}

@unittest.skipIf(SKIP_TEST, "Neuron dependencies are not available")
class TestNeuronRollingBatch(unittest.TestCase):

    def test_models(self):
        # === Preparation ===
        script_directory = os.path.dirname(os.path.abspath(__file__))
        relative_path = "../rolling_batch_test_scripts"
        new_path = os.path.normpath(
            os.path.join(script_directory, relative_path))
        sys.path.append(new_path)
        from djl_python.tests.rolling_batch_test_scripts.run_rolling_batch_alone import  init_rolling_batch, simulator

        # --- Models ---
        model_names = [
            "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
        ]

        # === Test ===
        for model_id in model_names:
            properties = {
                "tensor_parallel_degree": 1,
                "n_positions": "128",
                "rolling_batch": "tnx",
                "max_rolling_batch_size": 4,
                "model_loading_timeout": 3600,
                "model_id": model_id
            }

            # ===================== neuron-tnx ============================
            print('========== init inference ===========')
            input_str1 = [
                "Hello, my name is",  # 6
                "The president of the United States is",  # 8
                "The capital of France is",  # 6
                "The future of AI is"
            ]  # 7

            params1 = [{
                "max_new_tokens": 100,
                "do_sample": False,
            }.copy() for _ in range(len(input_str1))]

            batcher = init_rolling_batch("neuron", model_id, properties)
            simulator(batcher, input_str1, params1, [1, 1, 1, 1], 1)

            batcher = None
            import gc
            gc.collect()


if __name__ == '__main__':
    #unittest.main()
    c = TestNeuronRollingBatch()
    c.test_models()