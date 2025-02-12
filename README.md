# Fake News Detection Distilbert
This model was trained on 44,878 news articles from CLÉMENT BISAILLON's dataset on Kaggle. The goal is to classify fake news from real news. 

Dataset label structure:

0 : Fake News, 1 : Real News

## About Dataset

### ISOT Fake News detection dataset

Dataset separated in two files:

Fake.csv (23502 fake news article)
True.csv (21417 true news article)

Dataset columns:

Title: title of news article
Text: body text of news article
Subject: subject of news article
Date: publish date of news article

## About Model

Model Description: 
This model is a fine-tune checkpoint of distilbert-base-uncased-finetuned-sst-2-english, fine-tuned on ISOT Fake News Dataset.

## Fine-tuning hyper-parameters

`learning_rate = 2e-5
batch_size = 32
warmup = 600
max_seq_length = 128
num_train_epochs = 3.0`

# Sources
Dataset used: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Base Model (Distilbert): https://huggingface.co/distilbert/distilbert-base-uncased