import torch
import numpy as np 
from dataclasses import asdict, dataclass, field
from transformers import DataCollatorForSeq2Seq


# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
from typing import Dict, List, Optional

from transformers import (
    BatchEncoding,
    PreTrainedTokenizerBase,
)


@dataclass
class TaskDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
   def check_uniqueness(self, samples):
        assert len(np.unique(samples)) == 1 

   def __call__(self, features):
        tasks = [d.pop('task') for d in features]
        self.check_uniqueness(tasks)
        output = super().__call__(features)
        output["task"] = tasks[0]
        return output


def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids


@dataclass
class DataCollatorForT5MLM:
     # modified from https://github.com/huggingface/transformers/blob/v4.21.3/examples/flax/language-modeling/run_t5_mlm_flax.py
     """
     Data collator used for T5 span-masked language modeling.
     It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
     For more information on how T5 span-masked language modeling works, one can take a look
     at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
     or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
     Args:
          tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
               The tokenizer used for encoding the data.
          noise_density (:obj:`float`):
               The probability with which to (randomly) mask tokens in the input.
          mean_noise_span_length (:obj:`float`):
               The average span length of the masked tokens.
          input_length (:obj:`int`):
               The expected input length after masking.
          target_length (:obj:`int`):
               The expected target length after masking.
          pad_token_id: (:obj:`int`):
               The pad token id of the model
          decoder_start_token_id: (:obj:`int):
               The decoder start token id of the model
     """

     tokenizer: PreTrainedTokenizerBase
     noise_density: float
     mean_noise_span_length: float
     input_length: int
     target_length: int
     pad_token_id: int
     decoder_start_token_id: int

     def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:

          inputs = [e["source"] for e in examples]
          batch = self.tokenizer(inputs, max_length=self.input_length, padding=False, truncation=True, add_special_tokens=True, return_attention_mask=False)

          input_ids = batch["input_ids"]

          batch_size = len(input_ids)

          filtered_inputs = []
          filtered_labels = []

          for i in range(batch_size):
               array_input = np.expand_dims(np.array(input_ids[i]), 0)
               mask_indices = self.random_spans_noise_mask(array_input.shape[1])
               mask_indices = np.expand_dims(mask_indices, 0)
               labels_mask = ~mask_indices
               # input_ids_pad[i, :len(input_ids[i])] = input_ids[i]

               input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
               labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

               filtered_input = self.filter_input_ids(array_input, input_ids_sentinel)
               filtered_label = self.filter_input_ids(array_input, labels_sentinel)

               filtered_inputs.append(filtered_input)
               filtered_labels.append(filtered_label)

          max_input_length = max([inp.shape[1] for inp in filtered_inputs])
          max_target_length = max([inp.shape[1] for inp in filtered_labels])

          attention_mask = np.zeros((batch_size, max_input_length))
          input_ids_pad = np.zeros((batch_size, max_input_length), dtype=int)
          label_ids_pad = np.zeros((batch_size, max_target_length), dtype=int)

          for i in range(batch_size):
               input_ids_pad[i, :filtered_inputs[i].shape[1]] = filtered_inputs[i][0]
               attention_mask[i, :filtered_inputs[i].shape[1]] = 1
               label_ids_pad[i, :filtered_labels[i].shape[1]] = filtered_labels[i][0]

          label_ids_pad[label_ids_pad == self.pad_token_id] = -100

          batch["input_ids"] = input_ids_pad
          batch["attention_mask"] = attention_mask
          batch["labels"] = label_ids_pad

          tasks = [e.pop('task') for e in examples]
          self.check_uniqueness(tasks)

          batch["task"] = tasks[0]

          # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
          batch["decoder_input_ids"] = shift_tokens_right(
               batch["labels"], self.pad_token_id, self.decoder_start_token_id
          )


          batch["input_ids"] = torch.LongTensor(batch["input_ids"])
          batch["attention_mask"] = torch.FloatTensor(batch["attention_mask"])
          batch["labels"] = torch.LongTensor(batch["labels"])
          batch["decoder_input_ids"] = torch.LongTensor(batch["decoder_input_ids"])

          return batch

     def check_uniqueness(self, samples):
        assert len(np.unique(samples)) == 1 

     def create_sentinel_ids(self, mask_indices):
          """
          Sentinel ids creation given the indices that should be masked.
          The start indices of each mask are replaced by the sentinel ids in increasing
          order. Consecutive mask indices to be deleted are replaced with `-1`.
          """
          start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
          start_indices[:, 0] = mask_indices[:, 0]

          sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
          sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
          sentinel_ids -= mask_indices - start_indices

          return sentinel_ids

     def filter_input_ids(self, input_ids, sentinel_ids):
          """
          Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
          This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
          """
          batch_size = input_ids.shape[0]

          input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)

          # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
          # masked tokens coming after sentinel tokens and should be removed
          input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
          input_ids = np.concatenate(
               [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
          )
          return input_ids

     def random_spans_noise_mask(self, length):

          """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
          Noise mask consisting of random spans of noise tokens.
          The number of noise tokens and the number of noise spans and non-noise spans
          are determined deterministically as follows:
          num_noise_tokens = round(length * noise_density)
          num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
          Spans alternate between non-noise and noise, beginning with non-noise.
          Subject to the above restrictions, all masks are equally likely.
          Args:
               length: an int32 scalar (length of the incoming token sequence)
               noise_density: a float - approximate density of output mask
               mean_noise_span_length: a number
          Returns:
               a boolean tensor with shape [length]
          """

          orig_length = length

          num_noise_tokens = int(np.round(length * self.noise_density))
          # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
          num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
          num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

          # avoid degeneracy by ensuring positive number of noise spans
          num_noise_spans = max(num_noise_spans, 1)
          num_nonnoise_tokens = length - num_noise_tokens

          # pick the lengths of the noise spans and the non-noise spans
          def _random_segmentation(num_items, num_segments):
               """Partition a sequence of items randomly into non-empty segments.
               Args:
                    num_items: an integer scalar > 0
                    num_segments: an integer scalar in [1, num_items]
               Returns:
                    a Tensor with shape [num_segments] containing positive integers that add
                    up to num_items
               """
               mask_indices = np.arange(num_items - 1) < (num_segments - 1)
               np.random.shuffle(mask_indices)
               first_in_segment = np.pad(mask_indices, [[1, 0]])
               segment_id = np.cumsum(first_in_segment)
               # count length of sub segments assuming that list is sorted
               _, segment_length = np.unique(segment_id, return_counts=True)
               return segment_length

          noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
          nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

          interleaved_span_lengths = np.reshape(
               np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
          )
          span_starts = np.cumsum(interleaved_span_lengths)[:-1]
          span_start_indicator = np.zeros((length,), dtype=np.int8)
          span_start_indicator[span_starts] = True
          span_num = np.cumsum(span_start_indicator)
          is_noise = np.equal(span_num % 2, 1)

          return is_noise[:orig_length]