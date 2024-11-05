# Copyright 2023 by Aysun Can Türetken.
# All rights reserved.


import os
import numpy as np
import datasets




class DataGenerator():
    # The data generation class that loads multiple financial sentiment analysis
    # datasets and merges them into a training and test splits. It then writes
    # the splits into two separate folders for model fine-tuning and evaluation.
    def __init__(self, load_from_disk = False, dataset_path = '',
                 class_balancing=True, test_split = 0.1, seed = 273):
        self.class_balancing = class_balancing
        self.test_split = test_split
        self.seed = seed
        self.instruction = "Classify the financial sentiment in the following text into one of the three classes: negative, neutral, or positive. Give a single word answer (either negative, neutral or positive)."
        self.instruction_column_name = 'instruction'
        self.input_text_column_name = 'text'
        self.label_column_name = 'label'
        self.negative_label = 'negative'
        self.neutral_label = 'neutral'
        self.positive_label = 'positive'

        if load_from_disk:
            self.load_from_disk(dataset_path)
        else:
            self.financial_phrasebank_dataset = self.load_financial_phrasebank()
            self.twitter_financial_news_sentiment = self.load_twitter_financial_news_sentiment()
            self.financial_opinion_mining_and_question_answering = self.load_financial_opinion_mining_and_question_answering()
            self.news_with_gpt_instructions = self.load_news_with_gpt_instructions()
        
        if class_balancing:
            max_no_samples = max(self.financial_phrasebank_dataset['train'].num_rows, self.twitter_financial_news_sentiment['train'].num_rows, self.financial_opinion_mining_and_question_answering['train'].num_rows, self.news_with_gpt_instructions['train'].num_rows)
            self.financial_phrasebank_dataset['train'] = self.replicate_dataset(self.financial_phrasebank_dataset['train'], max_no_samples)
            self.twitter_financial_news_sentiment['train'] = self.replicate_dataset(self.twitter_financial_news_sentiment['train'], max_no_samples)
            self.financial_opinion_mining_and_question_answering['train'] = self.replicate_dataset(self.financial_opinion_mining_and_question_answering['train'], max_no_samples)
            self.news_with_gpt_instructions['train'] = self.replicate_dataset(self.news_with_gpt_instructions['train'], max_no_samples)
        
        self.dataset = datasets.DatasetDict({"train":datasets.concatenate_datasets([self.financial_phrasebank_dataset['train'], self.twitter_financial_news_sentiment['train'], self.financial_opinion_mining_and_question_answering['train'], self.news_with_gpt_instructions['train']]), "test":datasets.concatenate_datasets([self.financial_phrasebank_dataset['test'], self.twitter_financial_news_sentiment['test'], self.financial_opinion_mining_and_question_answering['test'], self.news_with_gpt_instructions['test']])})
        
        self.dataset['train'] = self.dataset['train'].shuffle(seed = seed)
        self.dataset['test'] = self.dataset['test'].shuffle(seed = seed)


    def replicate_dataset(self, dataset, target_no_samples):
        replication_factor = int(round(float(target_no_samples) / float(dataset.num_rows)))
        if replication_factor > 1:
            dataset = datasets.concatenate_datasets([dataset] * replication_factor)
        return dataset


    def load_financial_dataset(self, dataset_path, subset_str, split_strs, input_text_column_name, label_column_name, label_mapping_func, test_split):
        """
        Method to load and format a given financial news/tweet dataset.
        """

        # Load and concatenate datasets if there are multiple subsets.
        if isinstance(split_strs, list):
            if len(split_strs) == 1:
                df = datasets.load_dataset(dataset_path, subset_str)[split_strs[0]].to_pandas()
            else:
                ds = datasets.load_dataset(dataset_path, subset_str)
                ds_list = []
                for subset_str in split_strs:
                    ds_list.append(ds[subset_str])
                df = datasets.concatenate_datasets(ds_list).to_pandas()
        else:
            df = datasets.load_dataset(dataset_path, subset_str)[split_strs].to_pandas()

        # Get only the necessary columns.
        df = df[[input_text_column_name, label_column_name]]

        # Replace all the cells which contain empty strings with nans.
        df = df.replace(' ', np.nan)
        df = df.replace('', np.nan)
        # Remove all the rows which contain None instead of a valid string.
        df = df.dropna()

        # Map the labels and rename columns .
        if label_mapping_func is not None:
            df[label_column_name] = df[label_column_name].apply(label_mapping_func)
        df.rename(columns={input_text_column_name: self.input_text_column_name, label_column_name: self.label_column_name}, inplace=True)
        df[self.instruction_column_name] = self.instruction
        df = df[[self.instruction_column_name, self.input_text_column_name, self.label_column_name]]

        if test_split:
            return datasets.Dataset.from_pandas(df).train_test_split(test_size=self.test_split, seed=self.seed)
        else:
            return datasets.Dataset.from_pandas(df)


    def load_twitter_financial_news_sentiment(self):
        """
        Method to load and format the twitter financial news sentiment (TFNS) dataset.
        """

        label_mapping_func = lambda x:{0:self.negative_label,1:self.positive_label,2:self.neutral_label}[x]
        train_dataset = self.load_financial_dataset(dataset_path = "zeroshot/twitter-financial-news-sentiment",
                                         subset_str = None,
                                         split_strs = "train",
                                         input_text_column_name = "text",
                                         label_column_name = "label",
                                         label_mapping_func = label_mapping_func,
                                         test_split = False)
        test_dataset = self.load_financial_dataset(dataset_path = "zeroshot/twitter-financial-news-sentiment",
                                         subset_str = None,
                                         split_strs = "validation",
                                         input_text_column_name = "text",
                                         label_column_name = "label",
                                         label_mapping_func = label_mapping_func,
                                         test_split = False)
        return datasets.DatasetDict({"train":train_dataset, "test":test_dataset})

    def load_financial_phrasebank(self):
        """
        Method to load and format the financial phrasebank (FPB) dataset.
        """

        label_mapping_func = lambda x:{0:self.negative_label,1:self.neutral_label,2:self.positive_label}[x]
        return self.load_financial_dataset(dataset_path = "financial_phrasebank",
                                         subset_str = "sentences_50agree",
                                         split_strs = "train",
                                         input_text_column_name = "sentence",
                                         label_column_name = "label",
                                         label_mapping_func = label_mapping_func,
                                         test_split = True)


    def load_financial_opinion_mining_and_question_answering(self):
        """
        Method to load and format the financial opinion mining and question answering (FOMQA) dataset.
        """

        label_mapping_thr = 0.33
        label_mapping_func = lambda x: self.negative_label if x<=-label_mapping_thr else (self.positive_label if x>=label_mapping_thr else self.neutral_label)

        train_dataset = self.load_financial_dataset(dataset_path = "pauri32/fiqa-2018",
                                         subset_str = None,
                                         split_strs = ["train", 'validation'],
                                         input_text_column_name = "sentence",
                                         label_column_name = "sentiment_score",
                                         label_mapping_func = label_mapping_func,
                                         test_split = False)
        test_dataset = self.load_financial_dataset(dataset_path = "pauri32/fiqa-2018",
                                         subset_str = None,
                                         split_strs = "test",
                                         input_text_column_name = "sentence",
                                         label_column_name = "sentiment_score",
                                         label_mapping_func = label_mapping_func,
                                         test_split = False)
        return datasets.DatasetDict({"train":train_dataset, "test":test_dataset})

    def load_news_with_gpt_instructions(self):
        """
        Method to load and format the news with gpt instructions (NGI) dataset.
        """

        label_mapping_func = lambda x:{'strong positive':self.positive_label,'moderately positive':self.positive_label,'mildly positive':self.neutral_label,'neutral':self.neutral_label,'mildly negative':self.neutral_label,'moderately negative':self.negative_label,'strong negative':self.negative_label}[x]
        train_dataset = self.load_financial_dataset(dataset_path = "oliverwang15/news_with_gpt_instructions",
                                         subset_str = None,
                                         split_strs = "train",
                                         input_text_column_name = "news",
                                         label_column_name = "label",
                                         label_mapping_func = label_mapping_func,
                                         test_split = False)
        test_dataset = self.load_financial_dataset(dataset_path = "oliverwang15/news_with_gpt_instructions",
                                         subset_str = None,
                                         split_strs = "test",
                                         input_text_column_name = "news",
                                         label_column_name = "label",
                                         label_mapping_func = label_mapping_func,
                                         test_split = False)
        return datasets.DatasetDict({"train":train_dataset, "test":test_dataset})

    def get_combined_dataset(self):
        return self.dataset
    def get_financial_phrasebank_dataset(self):
        return self.financial_phrasebank_dataset
    def get_twitter_financial_news_sentiment_dataset(self):
        return self.twitter_financial_news_sentiment
    def get_financial_opinion_mining_and_question_answering_dataset(self):
        return self.financial_opinion_mining_and_question_answering
    def get_news_with_gpt_instructions_dataset(self):
        return self.news_with_gpt_instructions

    def save_to_disk(self, dataset_path):
        self.dataset.save_to_disk(dataset_path)
        self.financial_phrasebank_dataset.save_to_disk(os.path.join(os.path.dirname(dataset_path), 'financial_phrasebank_dataset'))
        self.twitter_financial_news_sentiment.save_to_disk(os.path.join(os.path.dirname(dataset_path), 'twitter_financial_news_sentiment'))
        self.financial_opinion_mining_and_question_answering.save_to_disk(os.path.join(os.path.dirname(dataset_path), 'financial_opinion_mining_and_question_answering'))
        self.news_with_gpt_instructions.save_to_disk(os.path.join(os.path.dirname(dataset_path), 'news_with_gpt_instructions'))

    def load_from_disk(self, dataset_path):
        self.dataset = datasets.load_from_disk(dataset_path)
        self.financial_phrasebank_dataset = datasets.load_from_disk(os.path.join(os.path.dirname(dataset_path),'financial_phrasebank_dataset'))
        self.twitter_financial_news_sentiment = datasets.load_from_disk(os.path.join(os.path.dirname(dataset_path), 'twitter_financial_news_sentiment'))
        self.financial_opinion_mining_and_question_answering = datasets.load_from_disk(os.path.join(os.path.dirname(dataset_path), 'financial_opinion_mining_and_question_answering'))
        self.news_with_gpt_instructions = datasets.load_from_disk(os.path.join(os.path.dirname(dataset_path), 'news_with_gpt_instructions'))
