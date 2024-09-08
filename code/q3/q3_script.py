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
import math

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
        
# download the model
fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('../cc.en.300.bin')
embedding_gen = FastTextEmbeddingGenerator()
embedding_gen.set_model(ft)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        # self.encoding = self.encoding.unsqueeze(0)  # Add a batch dimension

    def forward(self, x):
        seq_len = x.size(1)
        return self.encoding[:seq_len, :]

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
                if c > 97:
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
train_sentences, val_sentences, test_sentences = dataset.get_splits(val_size=VAL_SPLIT, test_size=TEST_SPLIT)

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

# add EOS token
# word2idx['<EOS>'] = len(word2idx)
# idx2word[len(idx2word)] = '<EOS>'
# dataset.vocab.append('<EOS>')
# EOS_IDX = word2idx['<EOS>']

vocab_size = len(word2idx)

print(f"Number of training sentences: {len(train_sentences)}")

class SequentialData(Dataset):
    def __init__(self, sentences, embedding_gen, word2idx, max_len, d_model):
        self.sentences = sentences
        self.embedding_gen = embedding_gen
        self.word2idx = word2idx
        self.max_len = max_len
        self.pad_token = '<PAD>'
        self.pad_idx = word2idx[self.pad_token]
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.d_model = d_model

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        try:
            sentence = self.sentences[idx]
            sentence = sentence[:self.max_len]  # Truncate sentence to max_len
            tokens = sentence.split()
            sentence_embedding = [self.embedding_gen.get_embedding(token) if token in self.word2idx else self.embedding_gen.get_embedding('<UNK>') for token in tokens]
            target_indices = [self.word2idx[token] if token in self.word2idx else self.word2idx['<UNK>'] for token in tokens]

            padding_len = self.max_len - len(tokens)
            if padding_len > 0:
                sentence_embedding.extend([np.zeros_like(sentence_embedding[0])] * padding_len)  # Pad embeddings with zero vectors
                target_indices.extend([self.pad_idx] * padding_len)  # Pad indices with pad_idx
            
            # convert sentence_embedding to tensor
            sentence_embedding = np.array(sentence_embedding)

            # print(f"Sentence Embedding Shape Before Anything new: {sentence_embedding.shape}")
            padding_mask = [0 if i >= len(tokens) else 1 for i in range(self.max_len)]

            # Padding mask is 0 for padded tokens and 1 for tokens that are part of the sequence
            # -- in alignment with the transformer paper

            sentence_embedding = torch.tensor(sentence_embedding).float()
            target_indices = torch.tensor(target_indices)
            
            # print(f"Sentence Embedding Shape: {sentence_embedding.shape}")
            # print(f"Target Indices Shape: {target_indices.shape}")
            # return sentence_embedding, target_indices, torch.tensor(padding_mask)
        
            positional_encoding = self.positional_encoding(sentence_embedding)  # Get positional encodings
            sentence_embedding = sentence_embedding + positional_encoding  # Fuse FastText and Positional Embeddings

            return np.array(sentence_embedding), np.array(target_indices), torch.tensor(padding_mask)
        
        except Exception as e:
            print(f"Error processing sequence: {self.sentences[idx]}")
            print(f"Error as {e}")
            return None

d_model = 300
train_dataset = SequentialData(train_sentences, embedding_gen, word2idx, dataset.max_sentence_length, d_model)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
print(dataset.max_sentence_length)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x, padding_mask, attn_mask):
        norm_x = self.norm1(x)
        attn_output, _ = self.multihead_attn(norm_x, norm_x, norm_x, attn_mask=attn_mask, key_padding_mask=padding_mask)
        x = attn_output + x
        norm_x = self.norm2(x)
        x = self.mlp(norm_x) + x
        return x

class Transformer(nn.Module):
    def __init__(self, num_emb, hidden_size=128, num_layers=3, num_heads=4):
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_size, num_emb)
        self.vocab_size = num_emb

    def generate_square_subsequent_mask(self, sz):
        # Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, tgt_mask, padding_mask):
        # x already contains the embeddings and positional encodings
        for block in self.blocks:
            x = block(x, padding_mask, attn_mask=tgt_mask)
        return self.fc_out(x)

print("Model Created")

def train(model, train_loader, optimizer, criterion, device, clip_grad=1.0):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target, padding_mask) in enumerate(progress_bar):
        gc.collect()
        torch.cuda.empty_cache()   
        
        data, target, padding_mask = data.to(device), target.to(device), padding_mask.to(device) 
        optimizer.zero_grad()
        
        input_seq = data[:, :-1, :]
        target_seq = target[:, 1:]

        # print(f"Input Sequence Shape: {input_seq.shape}")

        # padding mask is 0 for padded tokens and 1 for tokens that are part of the sequence
        tgt_key_padding_mask = (padding_mask[:, :-1] == 0)
        
        tgt_mask = model.generate_square_subsequent_mask(input_seq.size(1)).to(device)

        output = model(input_seq, tgt_mask=tgt_mask, padding_mask=tgt_key_padding_mask)
        output_reshape = output.view(-1, model.vocab_size)
        target_seq_reshape = target_seq.reshape(-1)

        loss = criterion(output_reshape, target_seq_reshape) 

        del data, target, padding_mask, input_seq, target_seq, tgt_key_padding_mask, tgt_mask, output, output_reshape, target_seq_reshape
        gc.collect()
        torch.cuda.empty_cache()

        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        gc.collect()
        torch.cuda.empty_cache()

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix({"Running Training Loss": total_loss / (batch_idx + 1)})
    
    return total_loss / len(train_loader)

# Evaluate on test set
val_dataset = SequentialData(val_sentences, embedding_gen, word2idx, dataset.max_sentence_length, d_model)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0

    progress_bar = tqdm(test_loader, desc="Validation")
    with torch.no_grad():
        for batch_idx, (data, target, padding_mask) in enumerate(progress_bar):
            data, target, padding_mask = data.to(device), target.to(device), padding_mask.to(device)

            input_seq = data[:, :-1, :]
            target_seq = target[:, 1:]

            tgt_key_padding_mask = (padding_mask[:, :-1] == 0)

            tgt_mask = model.generate_square_subsequent_mask(input_seq.size(1)).to(device)

            output = model(input_seq, tgt_mask=tgt_mask, padding_mask=tgt_key_padding_mask)

            output_reshape = output.view(-1, model.vocab_size)
            target_seq_reshape = target_seq.reshape(-1)
            loss = criterion(output_reshape, target_seq_reshape)

            total_loss += loss.item()
            progress_bar.set_postfix({"Running Validation Loss": total_loss / (batch_idx + 1)})
    
    return total_loss / len(test_loader)

hidden_size = 300
num_layers = 1
num_heads = 6
model = Transformer(vocab_size, hidden_size, num_layers, num_heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])

# Training loop
num_epochs = 5

# del train_dataset, val_dataset
gc.collect()
torch.cuda.empty_cache()

for epoch in range(num_epochs):
    loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss:.4f}")

torch.save(model, 'transformer_q3.pth')

# Evaluate on test set
test_dataset = SequentialData(test_sentences, embedding_gen, word2idx, dataset.max_sentence_length, d_model)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_loss = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")

def evaluate_sentence_wise(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    perplexities = []
    progress_bar = tqdm(test_loader, desc="In Progress Evaluation")
    with torch.no_grad():
        for batch_idx, (data, target, padding_mask) in enumerate(progress_bar):
            data, target, padding_mask = data.to(device), target.to(device), padding_mask.to(device)

            input_seq = data[:, :-1, :]
            target_seq = target[:, 1:]

            tgt_key_padding_mask = (padding_mask[:, :-1] == 0)

            tgt_mask = model.generate_square_subsequent_mask(input_seq.size(1)).to(device)

            output = model(input_seq, tgt_mask=tgt_mask, padding_mask=tgt_key_padding_mask)

            output_reshape = output.view(-1, model.vocab_size)
            target_seq_reshape = target_seq.reshape(-1)
            loss = criterion(output_reshape, target_seq_reshape)
            curr_perplexity = torch.exp(loss)
            perplexities.append(curr_perplexity)
            total_loss += loss.item()
            progress_bar.set_postfix({"Running Validation Loss": total_loss / (batch_idx + 1)})
    
    # convert perplexities to numpy array by first loading them to cpu
    perplexities = [perplexity.cpu().numpy() for perplexity in perplexities]
    return np.array(perplexities)

eval_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
eval_test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

train_perplexities = evaluate_sentence_wise(model, eval_train_loader, criterion, device)

test_perplexities = evaluate_sentence_wise(model, eval_test_loader, criterion, device)

print("All Done!")

# create a pair of sentence, perplexity for train and test set
train_sentence_perplexity = [(sentence, perplexity) for sentence, perplexity in zip(train_sentences, train_perplexities)]
test_sentence_perplexity = [(sentence, perplexity) for sentence, perplexity in zip(test_sentences, test_perplexities)]

# exclude all pairs where the train sentence is less than 2 tokens
train_sentence_perplexity = [pair for pair in train_sentence_perplexity if len(pair[0].split()) > 1]
test_sentence_perplexity = [pair for pair in test_sentence_perplexity if len(pair[0].split()) > 1]

train_perplexities_filtered = [pair[1] for pair in train_sentence_perplexity]
test_perplexities_filtered = [pair[1] for pair in test_sentence_perplexity]

avg_train_perplexity = np.mean(train_perplexities_filtered)
avg_test_perplexity = np.mean(test_perplexities_filtered)

from scipy.stats import trim_mean
trimmed_train_perplexity = trim_mean(train_perplexities_filtered, 0.1)
trimmed_test_perplexity = trim_mean(test_perplexities_filtered, 0.1)

median_train_perplexity = np.median(train_perplexities_filtered)
median_test_perplexity = np.median(test_perplexities_filtered)

print(f"Average Test Perplexity (Trimmed): {trimmed_test_perplexity}")
print(f"Average Train Perplexity (Trimmed): {trimmed_train_perplexity}")

print(f"Median Test Perplexity: {median_test_perplexity}")
print(f"Median Train Perplexity: {median_train_perplexity}")

# write the pairs to a file
with open('2021101068-LM3-train-perplexity.txt', 'w') as file:
    for sentence, perplexity in train_sentence_perplexity:
        file.write(f"{sentence}\t{perplexity.item()}\n")
    
    file.write('\n')
    file.write(f"Average Perplexity: {avg_train_perplexity}\n")
    file.write(f"Average Perplexity (Trimmed -- excluding 0.1% from both ends): {trimmed_train_perplexity}\n")
    file.write(f"Median Perplexity: {median_train_perplexity}\n")

with open('2021101068-LM3-test-perplexity.txt', 'w') as file:
    for sentence, perplexity in test_sentence_perplexity:
        file.write(f"{sentence}\t{perplexity.item()}\n")

    file.write('\n')
    file.write(f"Average Perplexity: {avg_test_perplexity}\n")
    file.write(f"Average Perplexity (Trimmed -- excluding 0.1% from both ends): {trimmed_test_perplexity}\n")
    file.write(f"Median Perplexity: {median_test_perplexity}\n")

print("All Done!")