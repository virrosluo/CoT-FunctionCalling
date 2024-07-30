import logging
import os
import sys
import wandb

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from trl import SFTTrainer
from datasets import load_dataset

from huggingface_hub import login

from arguments import *
from data_collator import *
from prompt_creator import *
from utils import *

from env import *

logger = logging.Logger("main")

def set_environment():
    seed = 123
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    login(HF_TOKEN)

def main():
    set_environment()
    
    parser = HfArgumentParser([LoRaArguments, QuantizationArguments, ModelArguments, TokenizerArguments, DatasetArguments, ExperimentArguments, TrainingArguments])
    
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        lora_args, quantize_args, model_args, tokenizer_args, dataset_args, expr_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        lora_args, quantize_args, model_args, tokenizer_args, dataset_args, expr_args, training_args = parser.parse_args_into_dataclasses()
        
    if 'wandb' in training_args.report_to:
        wandb.init(project=WANDB_PROJECT, entity=ENTITY_NAME)
        
# ---------------------------------------------------------- CHECKING OUTPUT DIR
    logger.info("Checking output directory of the model training")
    last_checkpoint = get_newest_checkpoint(training_args=training_args)
        
    if training_args.bf16:
        compute_dtype = torch.bfloat16
    elif training_args.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32
        
# ---------------------------------------------------------- LOAD MODEL CONFIGURATION 
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.revision,
        "token": model_args.token,
        "trust_remote_code": model_args.model_trust_remote_code
    }
    
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    
# ---------------------------------------------------------- LOAD DATASET
    datasets = load_dataset(
        path=dataset_args.dataset_path, 
        trust_remote_code=dataset_args.dataset_trust_remote_code
    )
    
# ---------------------------------------------------------- LOAD MODEL TOKENIZER
    tokenizer_kwargs = {
        "pretrained_model_name_or_path": model_args.model_name_or_path,
        "cache_dir": model_args.cache_dir,
        "use_fast": tokenizer_args.use_fast_tokenizer,
        "revision": model_args.revision,
        "token": model_args.token,
        "trust_remote_code": tokenizer_args.tokenizer_trust_remote_code,
        "padding_side": "right",
        "model_max_length": model_args.model_max_length
    }
    
    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
    
    # Setting the pad tokenizer to the specific tokenizer
    if getattr(tokenizer, "pad_token", None) is None:
        logger.info("Re-assigning the PAD TOKEN of TOKENIZER")
        if tokenizer_args.set_pad_token == "END":
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif tokenizer_args.set_pad_token == "UNK":
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            raise ValueError("Tokenizer have no PAD TOKEN. Affecting the DataCollator Process!")
        
# ---------------------------------------------------------- LOAD MODEL
    # Config Quantization
    if quantize_args.use_quantize:
        if quantize_args.load_in_nbit == '4':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=quantize_args.bnb_nbit_compute_dtype,
                bnb_4bit_quant_type=quantize_args.bnb_nbit_quant_type,
                bnb_4bit_use_double_quant=quantize_args.bnb_nbit_use_double_quant
            )
        elif quantize_args.load_in_nbit == '8':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            raise ValueError("The quantization bit does not support")
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        quantization_config=quantization_config,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.revision,
        token=model_args.token,
        trust_remote_code=model_args.model_trust_remote_code,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention_2 else "eager"
    )
    
    if quantize_args.use_quantize:
        model = prepare_model_for_kbit_training(
            model=model, 
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'reentrant': False}
        )
    
    if lora_args.use_peft:
        logger.info("Preparing LoRA")
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            target_modules=lora_args.target_modules,
        )
        
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        
    model.config.use_cache=False
    training_args.gradient_checkpointing_kwargs={"use_reentrant": False}
    
# ---------------------------------------------------------- TRAINER INITIALIZATION
    collator = DataCollatorForCompletionOnlyLM(
        response_start_template=expr_args.response_start_template,
        response_end_template=expr_args.response_end_template,
        tokenizer=tokenizer
    )
    
    rouge_metric = metrics_creator(
        tokenizer=tokenizer
    )

    logger.info("Preparing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets['train'],
        eval_dataset={"valid":datasets['validation'], "test":datasets['test']},
        peft_config=lora_config,
        formatting_func=create_prompt(tokenizer, conversation_creator=conversation_factory(exper_args=expr_args)),
        data_collator=collator,
        args=training_args,
        packing=True,
        max_seq_length=model_args.model_max_length,
        compute_metrics=metrics_creator(tokenizer=tokenizer, metric=rouge_metric),
        neftune_noise_alpha=expr_args.neftune_noise
    )
    
    logger.info("Getting processed test datasets from SFTTrainer")
    test_loader = trainer.eval_dataset.pop("test")
    
    if not training_args.do_eval:
        trainer.eval_dataset.pop("valid")

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint
        else:
            checkpoint = last_checkpoint
        logger.info("Training from {}".format("start" if checkpoint is None else checkpoint))
    
    model.print_trainable_parameters()
    
# ---------------------------------------------------------- TEST BEFORE TRAINING
    logger.info("TEST BEFORE TRAINING THE MODEL")
    with trainer.model.disable_adapter():
        logger.info(f"Validation set result: {trainer.evaluate()}")
        logger.info(f"Test set result: {trainer.predict(test_loader).metrics}")
        
# ---------------------------------------------------------- TRAINING MODEL
    logger.info("START TRAINING THE MODEL")
    trainer.train(checkpoint)
    
# ---------------------------------------------------------- TEST AFTER TRAINING
    logger.info("TEST AFTER TRAINING THE MODEL")
    logger.info(f"Validation set result: {trainer.evaluate()}")
    logger.info(f"Test set result: {trainer.predict(test_loader).metrics}")
        
# ---------------------------------------------------------- SAVE MODEL AND BOARD
    trainer.save_model()
    trainer.push_to_hub(token=HF_TOKEN)
    
    if 'wandb' in training_args.report_to:
        wandb.finish()
        
if __name__ == '__main__':
    main()