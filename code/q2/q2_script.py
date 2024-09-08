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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = '../../data/'

VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

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

# fasttext.util.download_model('en', if_exists='ignore')        
# os.remove('cc.en.300.bin.gz')
ft = fasttext.load_model('../cc.en.300.bin')
embedding_gen = FastTextEmbeddingGenerator()
embedding_gen.set_model(ft)

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

                # if c > 97:
                #     break

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

class SequentialData(Dataset):
    def __init__(self, sentences, embedding_gen, word2idx, max_len):
        self.sentences = sentences
        self.embedding_gen = embedding_gen
        self.word2idx = word2idx
        self.max_len = max_len
        self.pad_token = '<PAD>'
        self.pad_idx = word2idx[self.pad_token]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        try:
            sentence = self.sentences[idx]
            # convert to lowercase
            sentence = sentence.lower()
            sentence = sentence[:self.max_len]  # Truncate sentence to max_len
            tokens = sentence.split()

            sentence_embedding = [self.embedding_gen.get_embedding(token) if token in self.word2idx else self.embedding_gen.get_embedding('<UNK>') for token in tokens]
            target_indices = [self.word2idx[token] if token in self.word2idx else self.word2idx['<UNK>'] for token in tokens]

            # Padding
            padding_len = self.max_len - len(tokens)
            if padding_len > 0:
                sentence_embedding.extend([np.zeros_like(sentence_embedding[0])] * padding_len)  # Pad embeddings with zero vectors
                target_indices.extend([self.pad_idx] * padding_len)  # Pad indices with pad_idx
            
            return np.array(sentence_embedding), np.array(target_indices)
        
        except Exception as e:
            print(f"Error processing sequence: {self.sentences[idx]}")
            print(f"Error: {e}")
            return None
        
train_dataset = SequentialData(train_sentences, embedding_gen, word2idx, dataset.max_sentence_length)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print("DataLoader created.")

class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        logits = self.fc(lstm_out)
        return logits, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(1, batch_size, self.hidden_dim).zero_().to(device),
                weight.new(1, batch_size, self.hidden_dim).zero_().to(device))
    
embedding_dim = 300
hidden_dim = 300
model = LSTMModel(embedding_dim, hidden_dim, vocab_size)

model.to(device) # Move model to GPU if available
 
optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])

n_epochs = 5

torch.cuda.empty_cache() # Clear cache before training

print("Starting training...")

model.train()
for epoch in range(n_epochs):
    total_loss = 0

    print(f"Epoch {epoch+1}")
    print(f"Number of batches: {len(train_loader)}")

    hidden = model.init_hidden(batch_size)
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    pbar.set_description(f"Epoch {epoch+1}")

    for i, batch in pbar:

        sentence_embeddings, sentence_indices = batch

        sentence_embeddings, sentence_indices = sentence_embeddings.to(device), sentence_indices.to(device)
        hidden = tuple(h.detach() for h in hidden)
        inputs = sentence_embeddings[:, :-1, :] # Cut off last token
        targets = sentence_indices[:, 1:]  # Shift targets by one position
        del sentence_embeddings

        output, hidden = model(inputs, hidden)
        del inputs

        output = output.view(-1, vocab_size)  # Shape: [batch_size * seq_len, vocab_size]
        targets = targets.reshape(-1)  # Shape: [batch_size * seq_len]

        loss = criterion(output, targets)
        del sentence_indices, targets, output

        loss.backward()

        optimizer.step() 
        optimizer.zero_grad()

        total_loss += loss.item()

        pbar.set_postfix_str(f"Running Loss: {total_loss / (i+1):.4f}")

        del loss
        gc.collect()
        torch.cuda.empty_cache()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

print("Training complete.")

print("Final average loss: ", avg_loss)
print("Final perplexity: ", np.exp(avg_loss))

model_complete_path = os.path.join(DATA_DIR, 'lm_q2_modified_fix_2.pth')
print(f"Model saved at {model_complete_path}")

torch.save(model, model_complete_path)

model_complete_path = os.path.join(DATA_DIR, 'checkpoint_q2.pth')

# Evaluate model on validation set
def calculate_perplexity(loss):  # Calculate perplexity from loss value: [Source](https://hackernoon.com/crossentropy-logloss-and-perplexity-different-facets-of-likelihood)
    return np.exp(loss)

def evaluate_model(model, data_loader, vocab_size, criterion):

    model.eval()
    total_loss = 0
    all_perplexities = []
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():

        for batch in data_loader:

            sentence_embeddings, sentence_indices = batch
            sentence_embeddings, sentence_indices = sentence_embeddings.to(device), sentence_indices.to(device)

            hidden = tuple(h.detach() for h in hidden)
            inputs = sentence_embeddings[:, :-1, :]  # Cut off last token
            targets = sentence_indices[:, 1:]  # Shift targets by one position
            output, hidden = model(inputs, hidden)
            # print(f"Output shape before reshape: {output.shape}")  
            output = output.view(-1, vocab_size)  # Shape: [batch_size * seq_len, vocab_size]
            targets = targets.reshape(-1)  # Shape: [batch_size * seq_len]

            # print(f"Output  shape: {output.shape}")
            # print(f"Targets shape: {targets.shape}")
            
            loss = criterion(output, targets)

            print("Loss value for this batch: ", loss.item())

            total_loss += loss.item()

            perplexity = calculate_perplexity(loss.item())
            
            all_perplexities.append(perplexity)

    avg_loss = total_loss / len(data_loader)
    avg_perplexity = np.mean(all_perplexities)
    
    return all_perplexities, avg_perplexity

import copy
print(len(test_sentences))

save_test_sentences = copy.deepcopy(test_sentences)

test_dataset = SequentialData(test_sentences, embedding_gen, word2idx, dataset.max_sentence_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# model.load_state_dict(torch.load(model_path))
# Load the entire model from model_complete_path
# model = torch.load(model_complete_path)
# check on which device model is
if next(model.parameters()).is_cuda:
    print('Model is on GPU')
else:
    print('Model is on CPU')
    model.to(device)

model.eval()

all_perplexities, avg_perplexity = evaluate_model(model, test_loader, vocab_size, criterion)
print(f"Length of all perplexities: {len(all_perplexities)}")
print("Averge of all perplexities: ", np.mean(all_perplexities))
print(f"Average perplexity on test set: {avg_perplexity:.4f}")

from scipy.stats import trim_mean

class SequentialDataWithSentence(Dataset):
    def __init__(self, sentences, embedding_gen, word2idx, max_len):
        self.sentences = sentences
        self.embedding_gen = embedding_gen
        self.word2idx = word2idx
        self.max_len = max_len
        self.pad_token = '<PAD>'
        self.pad_idx = word2idx[self.pad_token]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        try:
            sentence = self.sentences[idx]
            # convert to lowercase
            sentence = sentence.lower()
            sentence = sentence[:self.max_len]  # Truncate sentence to max_len
            tokens = sentence.split()

            sentence_embedding = [self.embedding_gen.get_embedding(token) if token in self.word2idx else self.embedding_gen.get_embedding('<UNK>') for token in tokens]
            target_indices = [self.word2idx[token] if token in self.word2idx else self.word2idx['<UNK>'] for token in tokens]

            # Padding
            padding_len = self.max_len - len(tokens)
            if padding_len > 0:
                sentence_embedding.extend([np.zeros_like(sentence_embedding[0])] * padding_len)  # Pad embeddings with zero vectors
                target_indices.extend([self.pad_idx] * padding_len)  # Pad indices with pad_idx
            
            return np.array(sentence_embedding), np.array(target_indices), sentence
        
        except Exception as e:
            print(f"Error processing sequence: {self.sentences[idx]}")
            print(f"Error: {e}")
            return None
        
test_dataset = SequentialDataWithSentence(save_test_sentences, embedding_gen, word2idx, dataset.max_sentence_length)

eval_batch_size = 1

test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

model.eval()

def calculate_perplexity_per_sentence(model, data_loader, vocab_size, criterion):

    model.eval()
    total_loss = 0
    all_perplexities = []
    hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():

        for batch in data_loader:

            sentence_embeddings, sentence_indices, sentence = batch
            sentence_embeddings, sentence_indices = sentence_embeddings.to(device), sentence_indices.to(device)

            hidden = tuple(h.detach() for h in hidden)
            inputs = sentence_embeddings[:, :-1, :]  # Cut off last token
            targets = sentence_indices[:, 1:]  # Shift targets by one position
            output, hidden = model(inputs, hidden)
            output = output.view(-1, vocab_size)  # Shape: [batch_size * seq_len, vocab_size]
            targets = targets.reshape(-1)  # Shape: [batch_size * seq_len]
            
            loss = criterion(output, targets)

            print("Loss value for this batch: ", loss.item())

            perplexity = calculate_perplexity(loss.item())
            
            all_perplexities.append((perplexity, sentence))

    return all_perplexities

all_perplexities = calculate_perplexity_per_sentence(model, test_loader, vocab_size, criterion)

all_perplexities = [pair for pair in all_perplexities if len(pair[1][0].split()) > 1]

test_perplexities = [perplexity for perplexity, sentence in all_perplexities]

# exclude the entries with nan perplexity
test_perplexities = np.array(test_perplexities)
avg_perplexity = np.mean(test_perplexities)
avg_trimmed_perplexity = trim_mean(test_perplexities, 0.1)
median_perplexity = np.median(test_perplexities)

print(f"Average perplexity on test set: {avg_perplexity:.4f}")
print(f"Trimmed mean perplexity on test set: {avg_trimmed_perplexity:.4f}")
print(f"Median perplexity on test set: {median_perplexity:.4f}")

# Write the perplexities to a file
with open('2021101068-LM2-test-perplexity.txt', 'w') as f:
    for perplexity, sentence in all_perplexities:
        f.write(f"{sentence[0]}\t{perplexity}\n")

    f.write('\n')
    f.write(f"Average Perplexity: {avg_perplexity}\n")
    f.write(f"Average Perplexity (Trimmed -- excluding 0.1% from both ends): {avg_trimmed_perplexity}\n")
    f.write(f"Median Perplexity: {median_perplexity}\n")

print("Perplexities written to file.")

# Write the perplexities to a file for train set 
train_dataset = SequentialDataWithSentence(train_sentences, embedding_gen, word2idx, dataset.max_sentence_length)
train_loader = DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=False)

all_perplexities = calculate_perplexity_per_sentence(model, train_loader, vocab_size, criterion)
print(f"Length of all perplexities: {len(all_perplexities)}")
all_perplexities = [(perplexity, sentence) for perplexity, sentence in all_perplexities if len(sentence[0].split()) > 1]

train_perplexities = [perplexity for perplexity, sentence in all_perplexities]

with open('2021101068-LM2-train-perplexity.txt', 'w') as f:
    for perplexity, sentence in all_perplexities:
        f.write(f"{sentence[0]}\t{perplexity}\n")

    f.write('\n')
    f.write(f"Average Perplexity: {np.mean(train_perplexities)}\n")
    f.write(f"Average Perplexity (Trimmed -- excluding 0.1% from both ends): {trim_mean(train_perplexities, proportiontocut=0.1)}\n")
    f.write(f"Median Perplexity: {np.median(train_perplexities)}\n")

print("Perplexities written to file.")