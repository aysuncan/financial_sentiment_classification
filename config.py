# Copyright 2023 by Aysun Can Türetken.
# All rights reserved.

import os
import torch



################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Number of training epochs
num_train_epochs = 4

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True

# Batch size per GPU for training
per_device_train_batch_size = 2 # reduce batch size by 2x if out-of-memory
                                # error (see https://medium.com/@iamarunbrahma/fine-tuning-of-falcon-7b-large-language-model-using-qlora-on-mental-health-dataset-aa290eb6ec85)

# Batch size per GPU for evaluation
per_device_eval_batch_size = 16 # reduce batch size by 2x if out-of-memory error

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 2 # increase gradient accumulation steps by 2x
                                # if batch size is reduced

# Enable gradient checkpointing. This currently gives an error!
gradient_checkpointing = False

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 5e-5

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.01

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 1000

# Log every X updates steps
logging_steps = 1000

overwrite_output_dir = True

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
# Make sure to pass a correct value for max_seq_length as the default value
# will be set to min(tokenizer.model_max_length, 1024).
max_seq_length = 768

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

# Check GPU compatibility with bfloat16
if getattr(torch, bnb_4bit_compute_dtype) == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
