#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
# The below code is heavily inspired from Optimum Neuron under the following link:
# https://github.com/huggingface/optimum-neuron/blob/974f34336bb36b1b64890c191c558a1575372be7/optimum/neuron/generation/token_selector.py

import logging
import copy
from typing import Optional, Union, List, Tuple
import torch
from transformers.generation import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
    TopKLogitsWarper,
)
from transformers.generation.utils import GenerationMode
from transformers.generation import LogitsWarper
from transformers_neuronx.speculation import TokenAcceptor
from djl_python.transformers_neuronx_scheduler.slot import Slot
from transformers_neuronx.speculation import SpeculativeGenerator, DraftProvider, TokenAcceptor, DefaultTokenAcceptor


class FastTopKLogitsWarper(LogitsWarper):
    """Returns [batch_size, top_k] scores and indices instead of [batch_size, vocab_size] scores."""

    def __init__(self, top_k: int):
        self.top_k = top_k

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        return torch.topk(scores, top_k)


class TokenSelector:
    """Implements the token selection logic corresponding to a generation configuration.

    This class combines and uses the logits processors and stopping criterias implemented in
    the transformers library.

    The algorithm to select these objects is heavily inspired by the transformers `GenerationMixin.generate()`
    method, but the actual token selection methods are specific.

    The reason why this class does not inherit from `GenerationMixin` is because it does not
    include the code to produce the tokens logits.
    Separating the production of the tokens logits from the tokens selection allows this class
    to be used with different generation paradigms, either synchronously using a single `TokenSelector` in
    `GenerationMixin.generate()` or asynchronously using multiple `TokenSelector` inside an inference endpoint.

    The constructor of this class should not be called directly: instances should be obtained by
    calling `TokenSelector.create()`.
    """

    def __init__(
        self,
        mode: GenerationMode,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        eos_token_id: Union[List[int], int],
        pad_token_id: int,
        logits_warper: Optional[LogitsProcessorList] = None,
        seed: Optional[int] = 0,
    ):
        self.mode = mode
        self.logits_processor = logits_processor
        self.stopping_criteria = stopping_criteria
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.logits_warper = logits_warper
        if self.mode == GenerationMode.SAMPLE:
            assert len(self.logits_warper) > 0
            last_warper = self.logits_warper[-1]
            self.fast_topk = isinstance(last_warper, TopKLogitsWarper)
            if self.fast_topk:
                # Replace the last warping operation by a faster alternative
                self.logits_warper[-1] = FastTopKLogitsWarper(
                    last_warper.top_k)

    @classmethod
    def create(
        cls,
        input_ids: torch.Tensor,
        generation_config: GenerationConfig,
        model: GenerationMixin,
        max_seq_length: int,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        seed: Optional[int] = 0,
    ) -> "TokenSelector":
        r"""Creates the `TokenSelector` for a specific generation configuration.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            generation_config (`~transformers.generation.GenerationConfig`, *optional*):
                The generation configuration to parametrize the token selection.
            model (`~transformers.generation.GenerationMixin`):
                The model provides the internal helpers allowing to select the logits processors and stopping criterias.
            max_seq_length (`int`):
                The maximum number of input + generated tokens for this model. It depends on the model compilation parameters.
            stopping_criteria (`Optional[transformers.generation.StoppingCriteriaList], defaults to `None`):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config.
            seed(`Optional[int]`):
                The optional seed for sampling. Defaults to zero.
        Return:
            `torch.LongTensor`: A `torch.LongTensor` containing the selected tokens.
        """
        generation_config.validate()
        generation_config = copy.deepcopy(generation_config)

        unsupported_generation_flags = [
            "output_attentions",
            "output_hidden_states",
            "output_scores",
            "return_dict_in_generate",
        ]
        for flag in unsupported_generation_flags:
            if getattr(generation_config, flag, False):
                raise ValueError("{flag} is not supported for generation.")

        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids.shape[
                -1]

        min_length = generation_config.min_length
        if min_length > max_seq_length:
            raise ValueError(
                f"The minimum generation length ({min_length}) exceeds the model maximum sequence length ({max_seq_length})"
            )
        max_length = generation_config.max_length
        if max_length > max_seq_length:
            generation_config.max_length = max_seq_length

        # Instantiate transformers library processors and criterias
        logits_processor = model._get_logits_processor(
            generation_config,
            input_ids_seq_length=input_ids.shape[-1],
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
        )
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
        stopping_criteria = model._get_stopping_criteria(
            generation_config, stopping_criteria=stopping_criteria)

        # The generation requires special tokens
        eos_token_id = generation_config.eos_token_id
        # This is not supposed to happen for any of the models we support
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = eos_token_id if isinstance(
                eos_token_id, int) else eos_token_id[0]

        generation_mode = model._get_generation_mode(generation_config, None)
        if generation_mode not in [
                GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE
        ]:
            raise ValueError("Unsupported generation mode")

        logits_warper = None
        if generation_mode == GenerationMode.SAMPLE:
            logits_warper = model._get_logits_warper(generation_config)

        return cls(mode=generation_mode,
                   logits_processor=logits_processor,
                   stopping_criteria=stopping_criteria,
                   logits_warper=logits_warper,
                   eos_token_id=eos_token_id,
                   pad_token_id=generation_config.pad_token_id,
                   seed=seed)

    def select(self, input_ids: torch.LongTensor,
               logits: torch.Tensor) -> (torch.LongTensor, torch.Tensor):
        """Select the next tokens from the candidate logits.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation (not used in all generation modes).
            logits (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The logits corresponding to the generated tokens.

        Return:
            `torch.LongTensor`: A `torch.LongTensor` containing the selected tokens.
        """
        scores = self.logits_processor(input_ids, logits)
        logprobs = torch.log_softmax(scores, -1)
        if self.mode == GenerationMode.SAMPLE:
            next_ids = self._sample(scores)
        else:
            next_ids = torch.argmax(scores, dim=-1)
        next_logprobs = torch.gather(logprobs, 1, next_ids.view(-1,
                                                                1)).view(-1)
        return next_ids, next_logprobs

    def _sample(self, scores: torch.Tensor) -> torch.LongTensor:
        if self.fast_topk:
            # Get [batch_size, top_k] scores and indices instead of [batch_size, vocab_size] scores
            scores, next_token_indices = self.logits_warper(None, scores)
        else:
            scores = self.logits_warper(None, scores)

        # sample
        probs = torch.nn.functional.softmax(scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        if self.fast_topk:
            # Convert the topk relative tokens to actual vocabulary tokens
            next_tokens = torch.gather(next_token_indices, 1, next_tokens)
        return next_tokens.squeeze(1)

class LMIDraftModelForSpeculation(DraftProvider):
    """
    Standard Implementation of Draft model provider that auto-regressively speculates k tokens.
    """

    def __init__(self, model, **kwargs) -> None:
        self.model = model
        self.kwargs = kwargs

    def _context_block(self, input_ids, start_ids):
        """
        Run context encoding network of the given model.

        Args:
            input_ids: The initial input tokens passed to the model
            start_ids: The offset from the beginning of each input in a batch.

        Returns:
            token: predicted next token
            score: predicted token score
        """
        next_token_scores = self.model(input_ids, None, start_ids)
        inputs = torch.argmax(next_token_scores, dim=1, keepdim=True)
        return inputs, next_token_scores

    def __call__(
            self,
            input_ids: torch.Tensor,
            k: int,
            cache_ids: Optional[torch.Tensor] = None,
            start_ids: Optional[torch.Tensor] = None,
            slot: Slot = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform standard auto-regressive token generation using the draft model, to speculate k-tokens.

        Args:
            input_ids: Either context, next token, or draft tokens. shape=(batch, seqlen)
            k: The number of speculative tokens
            cache_ids: The positions in the KV cache that should be updated. shape=(seqlen,)
            start_ids: The offset from the beginning of each input in a batch. shape=(batch,)

        Returns:
            tokens: The next token prediction(s)
            probabilities: The next token probability(s)
        """
        start_len = 0
        if cache_ids:
            start_len = cache_ids.item()

        if start_len == 0:  # run context network as cache_id location starts from 0.
            return self._context_block(input_ids, start_ids)

        next_token_scores = self.model(input_ids, cache_ids, start_ids)

        scores = []
        tokens = []

        # Speculate k tokens in auto regressive mode.
        for cur_len in range(start_len, start_len + k):
            next_len = cur_len + 1
            inputs = torch.argmax(next_token_scores, keepdim=True, dim=1)

            scores.append(next_token_scores)
            tokens.append(inputs)

            if next_len >= start_len + k:
                break

            cache_ids = torch.as_tensor([next_len], dtype=torch.int32)
            next_token_scores = self.model(inputs, cache_ids, start_ids)

        return (
            torch.cat(tokens, dim=1),
            torch.cat(scores, dim=0)
        )


class LMITokenAcceptor(TokenAcceptor):
    """
    Optimized TokenAcceptor based on original DeepMind paper: https://arxiv.org/pdf/2302.01318.pdf
    """
    def __call__(
            self,
            draft_ids: torch.Tensor,
            draft_scores: torch.Tensor,
            target_scores: torch.Tensor,
            slot: Slot,
    ) -> torch.Tensor:
        draft_token_len, draft_vocab = draft_scores.shape
        target_token_len, target_vocab = target_scores.shape
        assert draft_vocab == target_vocab  # vocab size should be same
        assert draft_token_len + 1 == target_token_len  # target should include additional token predicted

        draft_probabilities = torch.softmax(draft_scores, dim=-1)
        target_probabilities = torch.softmax(target_scores, dim=-1)
        index = draft_ids.view(-1, 1)
        target_probs = torch.gather(target_probabilities[:-1], 1, index)
        draft_probs = torch.gather(draft_probabilities, 1, index)

        random = torch.rand(draft_probs.shape)
        ratio = torch.clamp(target_probs / draft_probs, max=1.0)
        accepted = torch.less(random, ratio)

        # Minimum will return the first occurrence of 0 or False (i.e. rejection)
        minimum = torch.min(accepted.view(torch.uint8), dim=0)
        value = minimum.values.item()
        index = minimum.indices.item()

        if value != 0: # If we didn't get a rejection this means all drafts were accepted
            next_token, next_log_probs = slot.select(draft_ids, target_probabilities[-1:])
            next_token_id = torch.LongTensor([[next_token]])
            return torch.cat((draft_ids, next_token_id), dim=1)
        else:
            prob_diff = target_probabilities[index:index + 1] - draft_probabilities[index: index + 1]
            prob_diff = torch.clamp(prob_diff, min=0.0)
            next_token, next_log_probs = slot.select(draft_ids[index - 1: index], prob_diff)
            #next_token, next_log_probs = slot.select(draft_ids[index - 1: index], target_probabilities[index:index + 1])
            next_token_id = torch.LongTensor([[next_token]])
            return torch.cat((draft_ids[:, :index], next_token_id), dim=1)


class LMIGreedyTokenAcceptor(TokenAcceptor):
    """
    Optimized TokenAcceptor based on original DeepMind paper: https://arxiv.org/pdf/2302.01318.pdf
    """
    def __call__(
            self,
            draft_ids: torch.Tensor,
            draft_scores: torch.Tensor,
            target_scores: torch.Tensor,
            slot: Slot,
    ) -> torch.Tensor:
        draft_token_len, draft_vocab = draft_scores.shape
        target_token_len, target_vocab = target_scores.shape
        assert draft_vocab == target_vocab  # vocab size should be same
        assert draft_token_len + 1 == target_token_len  # target should include additional token predicted

        target_probabilities = torch.softmax(target_scores, dim=-1)
        target_ids = torch.argmax(target_probabilities, dim=1)
        draft_ids = draft_ids.squeeze()

        # Minimum will return the first occurrence of 0 or False (i.e. rejection)
        index = torch.where(draft_ids != target_ids[:-1])[0]

        if len(index) == 0: # If we didn't get a rejection this means all drafts were accepted
            return torch.unsqueeze(target_ids, 0)
        else:
            return torch.unsqueeze(target_ids[:index[0]+1], 0)
