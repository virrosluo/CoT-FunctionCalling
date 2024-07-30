from transformers import DataCollatorForLanguageModeling
from typing import (
    Union,
    List,
    Dict,
    Any
)

import numpy as np
import warnings

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only calculated on the completion made by the assistant.

    Differ from origin DataCollatorForCompletionOnlyLM is: origin DataCollatorForCompletionOnlyLM cannot take out the part of the ASSISTANT.
    We need to mask some part that is belong to the ASSISTANT in the Conversation.

    Args:
        response_start_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '<ASSISTANT> Response ...'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.

        response_end_template (`Union[str, List[int]]`): the template form that indicates the end of the response, typically something like
            '<|endoftext|>'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.

        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_start_template: Union[str, List[int]],
        response_end_template: Union[str, List[int], None],
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.response_start_template = response_start_template
        if isinstance(response_start_template, str):
            # The user provides a string, must tokenize
            self.response_start_token_ids = self.tokenizer.encode(self.response_start_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_start_token_ids = response_start_template

        self.response_end_template = response_end_template
        if isinstance(response_end_template, str):
            # The user provides a string, must tokenize
            self.response_end_token_ids = self.tokenizer.encode(self.response_end_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_end_token_ids = response_end_template

        if not self.mlm and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            response_token_ids_start_idx = []
            response_token_ids_end_idx = []

            for idx in np.where(batch["labels"][i] == self.response_start_token_ids[0])[0]:
                # `response_start_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                if (
                    self.response_start_token_ids
                    == batch["labels"][i][idx : idx + len(self.response_start_token_ids)].tolist()
                ):
                    response_token_ids_start_idx.append(idx)

            if self.response_end_token_ids is not None:
                for idx in np.where(batch["labels"][i] == self.response_end_token_ids[0])[0]:
                    if (
                        self.response_end_token_ids == batch["labels"][i][idx : idx + len(self.response_end_token_ids)].tolist()
                    ):
                        response_token_ids_end_idx.append(idx)

            if response_token_ids_start_idx == []:
                warnings.warn(
                    f"Could not find response key `{self.response_start_template}` in the "
                    f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                    f"This instance will be ignored in loss calculation. "
                    f"Note, if this happens often, consider increasing the `max_seq_length`."
                )
                batch["labels"][i, :] = self.ignore_index
            else:
                previous_end_idx = 0
                for current_idx in range(len(response_token_ids_start_idx)):
                    start_idx = response_token_ids_start_idx[current_idx]
                    end_startTemplate = start_idx + len(self.response_start_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, previous_end_idx:end_startTemplate] = self.ignore_index
                    if current_idx < len(response_token_ids_end_idx):
                        # Incase finding the endpoint of the chat template => Ignore the endpoint token
                        end_idx = response_token_ids_end_idx[current_idx]
                        end_endTemplate = end_idx + len(self.response_end_token_ids)
                        batch["labels"][i, end_idx:end_endTemplate] = self.ignore_index
                        previous_end_idx = end_endTemplate
                    else:
                        break
                else: # Incase finding all response key and its endpoint => Ignore all tokens behind the last response
                    batch["labels"][i, previous_end_idx:] = self.ignore_index
        return batch