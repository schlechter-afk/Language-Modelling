print("Hello World!")

import re
import random
from sklearn.model_selection import train_test_split
import spacy
import fasttext.util
import gensim
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from gensim.models import FastText
import pickle
import os
import gc
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import trim_mean

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = '../../data/'

VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

print("Imports done")

class FastTextEmbeddingGenerator:
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model
    
    def get_embedding(self, word):
        if word in self.model:
            embedding = self.model[word]
            return embedding
        else:
            embedding = self.model.get_word_vector(word)
            return embedding
        
ft = fasttext.load_model('../cc.en.300.bin')
embedding_gen = FastTextEmbeddingGenerator()
embedding_gen.set_model(ft)

print("FastText model loaded")

class LanguageModelDataset:
    def __init__(self, file_path, chunk_size=100000):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        self.sentences = self._process_large_file()
        self.train_sentences = None
        self.val_sentences = None
        self.test_sentences = None
        # self.max_sentence_length = max(len(sentence.split()) for sentence in self.sentences)  # Calculate global max sentence length
        self.max_sentence_length = 80  # Calculate global max sentence length

    # @functools.lru_cache(maxsize=None)
    def _process_large_file(self):
        sentences = []
        c = 0
        with open(self.file_path, 'r', encoding='utf-8') as file:
            buffer = ""
            for line in file:

                if c > 100:
                    break

                line = line.strip()
                if line: 

                    if buffer:
                        buffer += " " + line
                    else:
                        buffer = line

                else: 

                    if buffer:
                        temp_sentences = buffer.split(".")
                        for sentence in temp_sentences:
                            sentence = sentence.strip()
                            sentence = re.sub(r"[^a-zA-Z0-9\s]+", '', sentence)
                            sentence = sentence.strip()
                            # preprocess text
                            preprocessed_text = self._preprocess_text(sentence)
                            if preprocessed_text:
                                c += 1
                                sentences.append(preprocessed_text)

                        buffer = ""

            if buffer:
                buffer = buffer.strip()
                buffer = re.sub(r"[^a-zA-Z0-9\s]+", '', buffer)
                temp_sentences = buffer.split(".")
                for sentence in temp_sentences:
                    sentence = sentence.strip()
                    preprocessed_text = self._preprocess_text(sentence)
                    if preprocessed_text:
                        sentences.append(preprocessed_text)
                # sentences.append(self._preprocess_text(buffer))

        # sentences = [sentence for sentence in sentences if sentence != ""]
        return sentences

    # @functools.lru_cache(maxsize=None)
    def _preprocess_text(self, text):
        doc = self.nlp(text)
        sentences = " ".join([token.text for token in doc])
        return sentences

    # @functools.lru_cache(maxsize=None)
    def get_splits(self, val_size=10000, test_size=20000):
        train_sentences, val_test_sentences = train_test_split(self.sentences, test_size=val_size+test_size, shuffle=False, random_state=42)
        test_size = test_size / (val_size + test_size)
        val_sentences, test_sentences = train_test_split(val_test_sentences, test_size=test_size, shuffle=False, random_state=42)
        self.train_sentences = train_sentences
        self.val_sentences = val_sentences
        self.test_sentences = test_sentences
        return train_sentences, val_sentences, test_sentences
    
    def build_vocab(self):
        vocab = set()
        for sentence in self.train_sentences:
            for word in sentence.split():
                vocab.add(word)
        self.vocab = list(vocab)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

file_path = DATA_DIR + 'Auguste_Maquet.txt'
dataset = LanguageModelDataset(file_path)
print(f"Total number of sentences: {len(dataset.sentences)}")
train_sentences, val_sentences, test_sentences = dataset.get_splits(val_size=VAL_SPLIT, test_size=TEST_SPLIT)

print(f"Length of train_sentences: {len(train_sentences)}")
print(f"Length of val_sentences: {len(val_sentences)}")
print(f"Length of test_sentences: {len(test_sentences)}")

dataset.build_vocab() # Build vocabulary

word2idx = dataset.word2idx
idx2word = dataset.idx2word

# add UNK token
word2idx['<UNK>'] = len(word2idx)
idx2word[len(idx2word)] = '<UNK>'
dataset.vocab.append('<UNK>')

# add PAD token
word2idx['<PAD>'] = len(word2idx)
idx2word[len(idx2word)] = '<PAD>'
dataset.vocab.append('<PAD>')

vocab_size = len(word2idx)

print(f"Number of training sentences: {len(train_sentences)}")

from tqdm import tqdm

def extract_data(sentences, word2idx, embedding_gen, n_gram=5):
    inputs, targets = [], []

    for sentence in tqdm(sentences, desc="Processing Sentences"):
        try:
            words = sentence.split()

            # Precompute embeddings for all words in the sentence
            embeddings = np.array([embedding_gen.get_embedding(word) for word in words], dtype=np.float32)
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
            embeddings.to(device)
            
            # Create sliding windows of 5-gram context
            for i in range(len(words) - n_gram):
                context_embeddings = embeddings[i:i+n_gram].view(-1)  # Flatten the 5-gram embeddings
                inputs.append(context_embeddings)
                
                target_word = words[i + n_gram]

                if target_word in word2idx:
                    target_idx = word2idx[target_word]
                else:
                    target_idx = word2idx['<UNK>']

                targets.append(target_idx)

        except Exception as e:
            print(f"Error processing sentence: {sentence}")
            print(e)
            continue
    
    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    inputs = torch.stack(inputs)
    targets = torch.tensor(targets).to(device)
    return inputs, targets

# Exclude the sentences that are shorter than 5 words
train_sentences = [sentence for sentence in train_sentences if len(sentence.split()) > 5]

train_inputs, train_targets = extract_data(train_sentences, word2idx, embedding_gen)

print(f"Train Inputs Shape: {train_inputs.shape}")
print(f"Train Targets Shape: {train_targets.shape}")

torch.cuda.empty_cache()
gc.collect()

class NeuralLanguageModel(nn.Module):
    def __init__(self, n_gram, vocab_size, embedding_dim=300, hidden_dim=300, dropout=0.2):
        super(NeuralLanguageModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * n_gram, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)  # Output is vocab size
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        # return self.softmax(x)

model = NeuralLanguageModel(n_gram=5, vocab_size=vocab_size, embedding_dim=300, hidden_dim=300)

print(f"Model initialized")

torch.cuda.empty_cache()
gc.collect()

val_sentences = [sentence for sentence in val_sentences if len(sentence.split()) > 5]
val_inputs, val_targets = extract_data(val_sentences, word2idx, embedding_gen)

print(f"Val Inputs Shape: {val_inputs.shape}")
print(f"Val Targets Shape: {val_targets.shape}")

torch.cuda.empty_cache()
gc.collect()

batch_size = 64
epochs = 10
learning_rate = 0.001

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # L2 regularization

print(f"Model moved to device")

def calculate_val_loss(model, val_inputs, val_targets, criterion):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(range(0, len(val_inputs), batch_size), desc="Validation Batches")
        for i in pbar:
            inputs = val_inputs[i:i+batch_size].to(device)
            targets = val_targets[i:i+batch_size].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            pbar.set_postfix({"Current Validation Loss": loss.item()})

    val_loss /= (len(val_inputs)//batch_size)

    return val_loss

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0
    pbar = tqdm(range(0, len(train_inputs), batch_size), desc="Training Batches")
    for i in pbar:
        inputs = train_inputs[i:i+batch_size].to(device)
        targets = train_targets[i:i+batch_size].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({"Current Training Loss": loss.item()})
    
    train_loss = running_loss / (len(train_inputs)//batch_size)
    val_loss = calculate_val_loss(model, val_inputs, val_targets, criterion)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

    torch.cuda.empty_cache()
    gc.collect()

print("Training complete")

torch.save(model, 'language_model_q1.pth')

torch.cuda.empty_cache()
gc.collect()

test_sentences = [sentence for sentence in test_sentences if len(sentence.split()) > 5]
test_inputs, test_targets = extract_data(test_sentences, word2idx, embedding_gen)

print(f"Test Inputs Shape: {test_inputs.shape}")
print(f"Test Targets Shape: {test_targets.shape}")

# Evaluate on test set

def calculate_test_loss(model, test_inputs, test_targets, criterion):
    test_loss = 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(range(0, len(test_inputs), batch_size), desc="Test Batches")
        for i in pbar:
            inputs = test_inputs[i:i+batch_size].to(device)
            targets = test_targets[i:i+batch_size].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            pbar.set_postfix({"Current Test Loss": loss.item()})
    
    # return test_loss / (len(test_inputs)//batch_size)

    if len(test_inputs) < batch_size:
        test_loss /= 1
    else:
        test_loss /= (len(test_inputs)//batch_size)

    return test_loss

test_loss = calculate_test_loss(model, test_inputs, test_targets, criterion)
print(f"Test Loss: {test_loss}")

torch.cuda.empty_cache()
gc.collect()

def calculate_perplexity(model, sentence, word2idx, embedding_gen, criterion):
    model.eval()
    words = sentence.split()
    # embeddings = torch.tensor([embedding_gen.get_embedding(word) for word in words], dtype=torch.float32).to(device)
    embeddings = np.array([embedding_gen.get_embedding(word) for word in words], dtype=np.float32)
    embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
    total_loss = 0
    with torch.no_grad():
        for i in range(len(words) - 5):
            context_embeddings = embeddings[i:i+5].view(-1)
            target_word = words[i+5]
            target_idx = word2idx.get(target_word, word2idx['<UNK>'])
            target = torch.tensor([target_idx]).to(device)
            context_embeddings = context_embeddings.unsqueeze(0)
            context_embeddings = context_embeddings.to(device)
            output = model(context_embeddings)
            loss = criterion(output, target)
            total_loss += loss.item()

    perplexity = np.exp(total_loss / (len(words) - 5))
    return perplexity

def compute_perplexity_average(model, sentences, word2idx, embedding_gen, criterion):
    perplexities = []
    for sentence in sentences:
        perplexity = calculate_perplexity(model, sentence, word2idx, embedding_gen, criterion)
        perplexities.append(perplexity)
    return np.mean(perplexities)

perplexities_train = []
perplexities_test = []

with open('2021101068-LM1-train-perplexity.txt.txt', 'w') as file:
    for sentence in tqdm(train_sentences, desc="Processing Sentences"):
        perplexity = calculate_perplexity(model, sentence, word2idx, embedding_gen, criterion)
        perplexities_train.append(perplexity)
        file.write(f"{sentence}\t{perplexity}\n")

    file.write('\n')
    file.write(f"Average Perplexity: {np.mean(perplexities_train)}")
    file.write(f"Average Perplexity (Trimmed -- excluding 0.1% from both ends): {trim_mean(perplexities_train, proportiontocut=0.1)}")
    file.write(f"Median Perplexity: {np.median(perplexities_train)}")

torch.cuda.empty_cache()
gc.collect()

with open('2021101068-LM1-test-perplexity.txt', 'w') as file:
    for sentence in tqdm(test_sentences, desc="Processing Sentences"):
        perplexity = calculate_perplexity(model, sentence, word2idx, embedding_gen, criterion)
        perplexities_test.append(perplexity)
        file.write(f"{sentence}\t{perplexity}\n")
    
    file.write('\n')
    file.write(f"Average Perplexity: {np.mean(perplexities_test)}")
    file.write(f"Average Perplexity (Trimmed -- excluding 0.1% from both ends): {trim_mean(perplexities_test, proportiontocut=0.1)}")
    file.write(f"Median Perplexity: {np.median(perplexities_test)}")

torch.cuda.empty_cache()
gc.collect()

def train_with_hyperparameters(dropout_rates, hidden_dims, optimizers, criterion, train_X, train_y, val_X, val_y, epochs=10, batch_size=64):
    results = []
    train_perplexities = []
    val_perplexities = []

    for dropout in dropout_rates:
        for hidden_dim in hidden_dims:
            for opt in optimizers:
                model = NeuralLanguageModel(n_gram=5, vocab_size=vocab_size, embedding_dim=300, hidden_dim=hidden_dim).to(device)
                if opt == optimizers[0]:
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)

                for epoch in range(epochs):
                    model.train()
                    for i in tqdm(range(0, len(train_X), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
                        inputs = train_X[i:i+batch_size].to(device)
                        targets = train_y[i:i+batch_size].to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                    
                train_perplexity = compute_perplexity_average(model, train_sentences, word2idx, embedding_gen, criterion)
                val_perplexity = compute_perplexity_average(model, val_sentences, word2idx, embedding_gen, criterion)
                train_perplexities.append(train_perplexity)
                val_perplexities.append(val_perplexity)
                print(f"Train Perplexity: {train_perplexity}, Val Perplexity: {val_perplexity}")
                if opt == optimizers[0]:
                    results.append((dropout, hidden_dim, "Adam", train_perplexity, val_perplexity))
                else:
                    results.append((dropout, hidden_dim, "SGD", train_perplexity, val_perplexity))

                del model, optimizer, train_perplexity, val_perplexity
                torch.cuda.empty_cache()
                gc.collect()
    
    return results

dropout_rates = [0.2, 0.3]
hidden_dims = [300, 400]
optimizers = [optim.Adam, optim.SGD]

results = train_with_hyperparameters(dropout_rates, hidden_dims, optimizers, criterion, train_inputs, train_targets, test_inputs, test_targets, epochs=10, batch_size=64)

print("Hyperparameter tuning complete")

results_df = pd.DataFrame(results, columns=["Dropout", "Hidden Dim", "Optimizer", "Test Perplexity"])
results_df.to_csv('hyperparameter_results.csv', index=False)

print("Best Hyperparameters")
print(results_df.loc[results_df['Test Perplexity'].idxmin()])
print("Worst Hyperparameters")
print(results_df.loc[results_df['Test Perplexity'].idxmax()])

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(results_df['Hidden Dim'].unique()))
width = 0.25

for i, (optimizer, group) in enumerate(results_df.groupby('Optimizer')):
    for j, dropout in enumerate(group['Dropout'].unique()):
        data = group[group['Dropout'] == dropout]
        ax.bar(x + i * width + j * width / len(group['Dropout'].unique()), data['Test Perplexity'], width / len(group['Dropout'].unique()), label=f'{optimizer}, Dropout={dropout}')

ax.set_xlabel('Hidden Dim')
ax.set_ylabel('Test Perplexity')
ax.set_title('Variations of Test Perplexity by Dropout, Hidden Dim, and Optimizer')

ax.set_xticks(x + width)
ax.set_xticklabels(results_df['Hidden Dim'].unique())

ax.legend(title='Optimizer and Dropout')

plt.savefig('test_perplexity_plot.png')

plt.show()