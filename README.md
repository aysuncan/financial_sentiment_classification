This repository contains the source code I implemented for the final project of the 'Essentials in Text and Speech Processing' class (Fall 2023). The project involves fine-tuning 7B-parameter general-purpose LLMs for financial sentiment classification task. The results demonstrate that leveraging an extensive training dataset with recent 7-billion-parameter LLMs achieves state-of-the-art sentiment classification accuracy. For technical details and quantitative results, please refer to the project report final_report.pdf.

# Steps to Reproduce the Results

The results can be reproduced by either following the steps below or running the Jupyter notebook complete_steps.ipynb. These steps require a google drive to be available with sufficient space. Please replace them with your local drive paths if you want to use local storage.  


## Requirements

```console
pip install -q pipdeptree accelerate==0.22.0 peft==0.5.0 bitsandbytes==0.41.1 transformers==4.34.0 trl==0.7.2 sentencepiece==0.1.99
```

## Dataset Download and Pre-processing

This downloads and splits the four financial datasets mentioned in the report.

```console
python3 create_datasets.py '/content/gdrive/My Drive/Colab Notebooks/datasets/combined_dataset'
```


## Fine-Tuning

These commands download and fine-tune the Mistral-7B-OpenOrca and Llama-2-7b-chat-hf models on the training subset of the datasets.

```console
python3 train.py '/content/gdrive/My Drive/Colab Notebooks/datasets/combined_dataset' 'Open-Orca/Mistral-7B-OpenOrca' './results'
python3 train.py '/content/gdrive/My Drive/Colab Notebooks/datasets/combined_dataset' 'NousResearch/Llama-2-7b-chat-hf' './results'
```

## Tests with the Original Models

These are to perform inference with the original Mistral-7B-OpenOrca and Llama-2-7b-chat-hf models on the test subset of the datasets.

```console
python3 test.py '/content/gdrive/My Drive/Colab Notebooks/datasets/combined_dataset' 'Open-Orca/Mistral-7B-OpenOrca'
python3 test.py '/content/gdrive/My Drive/Colab Notebooks/datasets/combined_dataset' 'NousResearch/Llama-2-7b-chat-hf'
```

## Tests with the Fine-Tuned Models

Finally, these commands allow running inference with the fine-tuned models on the same test subset of the datasets.

```console
python3 test.py '/content/gdrive/My Drive/Colab Notebooks/datasets/combined_dataset' '/content/gdrive/My Drive/Colab Notebooks/results/llama_finetuned/'
python3 test.py '/content/gdrive/My Drive/Colab Notebooks/datasets/combined_dataset' '/content/gdrive/My Drive/Colab Notebooks/results/mistral_finetuned/'
```
