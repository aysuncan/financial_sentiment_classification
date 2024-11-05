# Copyright 2023 by Aysun Can Türetken.
# All rights reserved.

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback

from data_generator import DataGenerator
from config import *


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
       text = f"### INSTRUCTION: {example['instruction'][i]}\n ### TEXT: {example['text'][i]}\n ### ANSWER: {example['label'][i]}"
       output_texts.append(text)
    return output_texts

if __name__ == '__main__':

    no_arguments = len(sys.argv)
    if no_arguments < 4:
        print("Usage:", sys.argv[0], "<dataset folder path to be loaded>   <huggin-face model name to use as the base model>   <output directory path for storing the checkpoints and the final fine-tuned model>")
        exit()
    dataset_path = sys.argv[1]
    model_name = sys.argv[2]
    output_dir = sys.argv[3] # Output directory where the model predictions and checkpoints will be stored
    
    print("Loading the dataset ...")
    generator = DataGenerator(load_from_disk=True, dataset_path=dataset_path)
    
    # Get the training and test datasets.
    train_dataset = generator.get_combined_dataset()['train']
    test_dataset = generator.get_combined_dataset()['test']
    test_dataset = datasets.Dataset.from_pandas(test_dataset.to_pandas().head(test_dataset.num_rows//5)) # Use only 20 percent of the test dataset for printing the test loss during training.

    # Fine-tuned model path.
    finetuned_model_path = os.path.join(output_dir, model_name)

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
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    if use_minstral:
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules = ["q_proj", "v_proj"],
        )
    else:
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            r=lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Print traineble parameters.
    get_peft_model(model, peft_config).print_trainable_parameters()

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        do_eval=True,
        eval_steps=logging_steps,
        evaluation_strategy="steps",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard",
        load_best_model_at_end = True,
        #push_to_hub=True,
    )


    # We use the last three lines instead of the first two ones due to the following potential issue mentioned at: https://huggingface.co/docs/trl/sft_trainer#using-tokenids-directly-for-responsetemplate
    # response_template = " ### ANSWER:"
    # data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    response_template_with_context = "\n ### ANSWER:"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer)

    if gradient_checkpointing:
      model.gradient_checkpointing_enable()

    writer = SummaryWriter()

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
       model=model,
       train_dataset=train_dataset,
       eval_dataset=test_dataset,
       peft_config=peft_config,
       formatting_func=formatting_prompts_func, # DON'T USE THE FOLLOWING AS IT OVERRIDES THE COLLATOR: dataset_text_field="instruction",
       data_collator=data_collator,
       tokenizer=tokenizer,
       max_seq_length=max_seq_length,
       args=training_arguments,
       packing=packing,
       callbacks=[TensorBoardCallback(writer)],
    )

    # Train model
    trainer.train()

    writer.close()

    # Save trained model
    trainer.model.save_pretrained(finetuned_model_path)

    trainer.push_to_hub()
