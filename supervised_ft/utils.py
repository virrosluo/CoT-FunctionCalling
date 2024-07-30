import numpy as np
import os
import logging

from evaluate import load

from transformers.trainer_utils import get_last_checkpoint

import shutil

import nltk
nltk.download('punkt')

logger = logging.Logger("Utils")

def metrics_creator(tokenizer):
    metric=load('rouge')
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        pred_string = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        label_string = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in pred_string]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in label_string]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}
    return compute_metrics

def get_newest_checkpoint(training_args):
    '''
        Getting the last checkpoint on the output path dir that we are training
    '''
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        
        if last_checkpoint is None:
            logger.warning("Removing the current `training.args.output_dir` because cannot getting the last checkpoint from existing output directory")
            shutil.rmtree(training_args.output_dir)
            
        elif last_checkpoint is not None and training_args.resume_from_checkpoint:
            logger.warning("Continue training on last checkpoint althought `training_args.resume_from_checkpoint` is None. Set `overwrite_output_dir` = True for training from start")
    
    return last_checkpoint