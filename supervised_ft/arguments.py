from dataclasses import dataclass, field
from typing import (
    Optional,
    Union,
    Tuple
)

@dataclass
class LoRaArguments:
    use_peft: bool = field(
        default=True,
        metadata={
            'help': 'use peft for efficiency tranining'
        }
    )
    
    merge_adapter: bool = field(
        default=False,
        metadata={
            'help': 'merging the adapter to the model at the end'
        }
    )
    
    target_modules: Union[Tuple[str], str] = field(
        default=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"),
    )
    
    lora_r: int = field(
        default=8,
    )
    
    lora_dropout: float = field(
        default=0.1,
    )
    
    lora_alpha: float = field(
        default=16
    )
    
    lora_bias: str = field(
        default="none",
    )
    
@dataclass
class QuantizationArguments:
    use_quantize: bool = field(
        default=True
    )
    
    load_in_nbit: Optional[str] = field(
        default="4",
        metadata={
            'help': 'Load the original model into 4 bit quantization',
            'choices': ['4', '8']
        }
    )
    
    bnb_nbit_use_double_quant: bool = field(
        default=True,
        metadata={
            'help': 'Use double quantization for quantizing'
        }
    )
    
    bnb_nbit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={
            'help': 'Which quantization strategy will be used',
            'choices': ['nf4', 'nf8']
        }
    )
    
    bnb_nbit_compute_dtype: str = field(
        default="bfloat16",
        metadata={
            'help': 'Which floating point number will be use for quantization',
            'choices': ['bfloat16', 'float16', 'float32']
        }
    )

@dataclass
class TokenizerArguments:
    set_pad_token: Optional[str] = field(
        default='none',
        metadata={
            'help': 'Reassign the pad token into tokenizer',
            'choices': ['none', 'END', 'UNK']
        }
    )
    
    use_fast_tokenizer: bool = field(
        default=True
    )
    
    tokenizer_trust_remote_code: bool = field(
        default=False
    )

@dataclass
class DatasetArguments:
    dataset_path: str = field(
        default=None,
        metadata={
            'help': 'Dataset path on huggingface to clone down'
        }
    )
    
    dataset_trust_remote_code: bool = field(
        default=False
    )

@dataclass
class ExperimentArguments:
    
    experiment: str = field(
        default=None,
        metadata={
            'help': 'Which experiment that we are doing',
            'choices': ['mistral-cot', 'vistral-fc', 'vistral-fc-cot']
        }
    )
    
    response_start_template: Tuple[int] = field(
        default=None,
        metadata={
            'help': 'The list of token represent for the start of the assistant response',
            'choices': """Current choices is:
            [733, 28748, 16289, 28793] ## <=> [/INST] of Mistral v0.1 7B and v0.2 7B Instruct
            [28705, 38368, 28705] ## <=> [/INST] of Vistral 7B chat            
            """
        }
    )
    
    response_end_template: Tuple[int] = field(
        default=None,
        metadata={
            'help': 'The list of token represent for the end of the assistant response',
            'choices': """Current choices is: None          
            """
        }
    )
    
    neftune_noise: Optional[float] = field(
        default=5,
        metadata={
            'help': 'Adding neftune noise alpha in the embedding output for avoiding overfitting'
        }
    )

@dataclass 
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None
    )
        
    model_max_length: int = field(
        default=None,
        metadata={
            'help': 'Max token of input and output'
        }
    )
    
    revision: str = field(
        default=None,
        metadata={
            'help': 'Choosing the right repo for downloading the model weight'
        }
    )
    
    cache_dir: Optional[str] = field(
        default='.',
        metadata={
            'help': 'Directory path that you want to store the pretrained models downloaded from huggingface hub'
        }
    )
    
    token: bool = field(
        default=False,
        metadata={
            'help': 'Using the account token for accessing model'
        }
    )
    
    model_trust_remote_code: bool = field(
        default=False,
        metadata={
            'help': 'If setting to true, we will allow the model from huggingface change the code of our library, otherwise not'
        }
    )
    
    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={
            'help': "Override the default `torch.dtype` and load the model using this dtype. If `auto` it will use the model's weights default",
            'choices': ["auto", "bfloat16", "float16", "float32"]
        }
    )
    
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            'help': 'Reduce CPU and MEM when loading the model'
        }
    )
    
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={
            'help': 'use flash attention v2 for faster training'
        }
    )