# Copyright 2023 by Aysun Can Türetken.
# All rights reserved.

import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from data_generator import DataGenerator
from config import *


def load_model_and_tokenizer(online_model_name_or_offline_path):
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        online_model_name_or_offline_path,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = True
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(online_model_name_or_offline_path,
                                             trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    return model, tokenizer



def evaluate_and_compute_metrics_on_dataset(ds, no_pred_tokens):
    response_template_with_context = "\n ### ANSWER:"
    gt_classes = ['negative','neutral','positive']
    label_to_indx = {gt_classes[0]:0,gt_classes[1]:1,gt_classes[2]:2}
    invalid_class_indx = 3 # class 3 corresponds to the invalid class. These are for predictions (words) that are not in the allowable ground truth classes. They are neither 'negative', 'neutral' or 'positive'.
    no_samples = len(ds['instruction'])
    predictions = [invalid_class_indx] * no_samples
    gts = [0] * no_samples

    for i in tqdm(range(no_samples)):
        inputs = tokenizer(f"### INSTRUCTION: {ds['instruction'][i]}\n ### TEXT: {ds['text'][i]}\n ### ANSWER: ", return_tensors="pt", max_length=max_seq_length)
        outputs = model.generate(**inputs, max_length=len(inputs['input_ids'][0]) + no_pred_tokens, pad_token_id = tokenizer.eos_token_id)
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        gt = ds['label'][i]
        gts[i] = label_to_indx[gt]
        if answer.find(response_template_with_context) > 0: # do we have a sample which is not truncated? If not, do not consider this sample. This should never happen for any of our datasets as the max_seq_length is chosen large enough to avoid any truncation.
            prediction = answer[answer.find(response_template_with_context) + len(response_template_with_context):]
            prediction = prediction.lower()
            not_found_corresp_gt = True
            for c in gt_classes:
                if c in prediction:
                    predictions[i] = label_to_indx[c]
                    not_found_corresp_gt = False
                    break
            if not_found_corresp_gt:
                predictions[i] = invalid_class_indx

            #print("SAMPLE ", i, " LABEL: ", gt, gts[i], " PREDICTION: ", prediction, predictions[i])

    return gts, predictions




if __name__ == '__main__':
    no_arguments = len(sys.argv)
    if no_arguments < 3:
        print("Usage:", sys.argv[0], "<dataset folder path to be loaded> <model path to be loaded from a checkpoint or hugging-face model name>")
        exit()
    dataset_path = sys.argv[1]
    model_path = sys.argv[2]
    
    # For base models that are not pre-trained, we let them to generate more than one
    # token (also, word in our use-case) in case they generate a sentence instead of a
    # single word as instructed.
    if not os.path.isdir(model_path):
        print("Loading the hugging face model:", model_path)
        no_prediction_tokens = 10
    else:
        print("Loading the model from the checkpoint:", model_path)
        no_prediction_tokens = 1
        
    # Load the model.
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load the test datasets.
    generator = DataGenerator(load_from_disk=True, dataset_path=dataset_path)
    test_datasets = [generator.get_financial_phrasebank_dataset()['test'],
                    generator.get_twitter_financial_news_sentiment_dataset()['test'],
                    generator.get_financial_opinion_mining_and_question_answering_dataset()['test'],
                    generator.get_news_with_gpt_instructions_dataset()['test']]
    test_dataset_names = ['Financial Phrasebank Dataset',
                            'Twitter Financial News Sentiment Dataset',
                            'Financial Opinion Mining and_Question Answering Dataset',
                            'News with GPT Instructions Dataset']

    # Evaluate the model on the test datasets.
    for i in range(len(test_datasets)):
        print("Processing", test_dataset_names[i], '...')
        gts, predictions = evaluate_and_compute_metrics_on_dataset(test_datasets[i], no_prediction_tokens)
        precision, recall, fscore, support = score(gts, predictions)
        accuracy = accuracy_score(gts, predictions)
        conf_mat = confusion_matrix(gts, predictions)

        if 3 in np.unique(predictions):
            print(classification_report(gts, predictions, target_names=['negative','neutral','positive','undefined']))
        else:
            print(classification_report(gts, predictions, target_names=['negative','neutral','positive']))
        print('accuracy: {}'.format(accuracy))
        # print('precision: {}'.format(precision))
        # print('recall: {}'.format(recall))
        # print('fscore: {}'.format(fscore))
        # print('support: {}'.format(support))
        print("Confusion Matrix (Columns are predictions)")
        print("=======================================")
        print(conf_mat)
        print("=======================================")
