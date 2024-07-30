from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
)

from typing import (
    Callable,
    Union
)

def conversation_factory(exper_args):
    if exper_args.experiment == 'mistral-cot':
        return mistral_cot_conversation
    elif exper_args.experiment == 'vistral-fc':
        return vistral_fc_conversation
    elif exper_args.experiment == 'vistral-fc-cot':
        return vistral_fc_cot_conversation
    else:
        raise ValueError("Experiment choices in model arguments does not support")

def create_prompt(
    tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast], 
    conversation_creator: Callable):
    
    return lambda sample: tokenizer.apply_chat_template(
        conversation=conversation_creator(sample), 
        tokenize=False, 
        add_generation_prompt=True
    )

def mistral_cot_conversation(sample):
    system_message = "Let's think step by step to answer this question"
    conversation = [
        {"role":"user", "content": f"{system_message}: {sample['source']}"},
        {"role":"assistant", "content": f"{sample['rationale']}\nAnswer: {sample['target']}"}
    ]
    return conversation

def vistral_fc_conversation(sample):
    def preprocess_string(input_str: str) -> str:
        input_str = input_str.strip("\n")
        input_str = input_str.strip()
        return input_str
    
    system_message = f"Đưa ra json format cho các function calling với function description:\n{preprocess_string(sample['functionList'])}"
    assistant_message = f"Answer:{preprocess_string(sample['assistantResponse'])}"
    conversation = [
        {"role":"system", "content":system_message},
        {"role":"user", "content":f"{preprocess_string(sample['userPrompt'])}"},
        {"role":"assistant", "content":assistant_message}
    ]
    return conversation
    
def vistral_fc_cot_conversation(sample):
    def preprocess_string(input_str:str) -> str:
        input_str = input_str.strip("\n")
        input_str = input_str.strip()
        return input_str
    
    system_message = f"Suy nghĩ từng bước một để đưa ra json format cho các function calling với function description:\n{preprocess_string(sample['functionList'])}"
    assistant_message = f"{preprocess_string(sample['rationale'])}\nAnswer:{preprocess_string(sample['assistantResponse'])}"
    conversation = [
        {"role":"system", "content":system_message},
        {"role":"user", "content":f"{preprocess_string(sample['userPrompt'])}"},
        {"role":"assistant", "content":assistant_message}
    ]
    return conversation