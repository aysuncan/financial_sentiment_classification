# Copyright 2023 by Aysun Can Türetken.
# All rights reserved.

import os
import sys
from data_generator import DataGenerator


if __name__ == '__main__':

    no_arguments = len(sys.argv)
    if no_arguments < 2:
        print("Usage:", sys.argv[0], "<dataset folder path to be loaded ot created>")
        exit()
    dataset_path = sys.argv[1]
    
    print("Loading/Creating the dataset ...")
    
    generator = DataGenerator(load_from_disk=False, dataset_path=dataset_path)
    generator.save_to_disk(dataset_path)

    print("Done!")

    print("Summary of the Financial Phrasebank Dataset (FPB): ")
    print(generator.get_financial_phrasebank_dataset())
    print("Summary of the Twitter Financial News Sentiment Dataset (TFNS): ")
    print(generator.get_twitter_financial_news_sentiment_dataset())
    print("Summary of the Financial Opinion Mining and Question Answering Dataset (FOMQA): ")
    print(generator.get_financial_opinion_mining_and_question_answering_dataset())
    print("Summary of the News with GPT Instructions Dataset (NGI): ")
    print(generator.get_news_with_gpt_instructions_dataset())
    print("Summary of the Combined Dataset: ")
    print(generator.get_combined_dataset())
