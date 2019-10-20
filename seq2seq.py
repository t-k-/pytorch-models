import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import jieba
import nltk

debug = 0

stemmer = nltk.stem.SnowballStemmer('english')
# nltk.download('punkt')

def read_lines(file_path):
    fh = open(file_path, encoding='utf-8')
    content = fh.read().strip()
    content = content.lower()
    lines = content.split('\n')
    fh.close()
    return lines

zh_lines = read_lines('dataset/chinese.txt')
en_lines = read_lines('dataset/english.txt')

SOS_idx = 0
EOS_idx = 1

class BoW:
    def __init__(self):
        self.word2index = {}
        self.index2word = {SOS_idx: "SOS", EOS_idx: "EOS"}
        self.n_words = 2

    def addWords(self, array):
        for word in array:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

bow = [BoW(), BoW()]

MAX_LENGTH = 30
all_pairs = []

tot_lines = len(en_lines)

for l, pair in enumerate(zip(en_lines, zh_lines)):
    en_sentence, zh_sentence = pair
    en_words = [stemmer.stem(w) for w in nltk.word_tokenize(en_sentence)]
    if len(en_words) >= MAX_LENGTH:
        continue
    if debug and l > 300:
        break
    print('line %u %u' % (l, tot_lines), end='\r')
    zh_words = [w for w in jieba.cut(zh_sentence, cut_all=False)]
    # print(en_sentence)
    # print(en_words)
    # print(u" ".join(zh_words))
    bow[0].addWords(en_words)
    bow[1].addWords(zh_words)
    all_pairs.append((en_words, zh_words))

print('%u pairs of training sentenses' % len(all_pairs))
print('%u total of English words' % bow[0].n_words)
print('%u total of Chinese words' % bow[1].n_words)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensorFromWords(bow, arr):
    indexes = [bow.word2index[word] for word in arr]
    indexes.append(EOS_idx)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromWords(bow[0], pair[0])
    label_tensor = tensorFromWords(bow[1], pair[1])
    return (input_tensor, label_tensor)

# print(all_pairs[0])
# print(tensorsFromPair(all_pairs[0]))

hidden_size = 256
encoder = EncoderRNN(bow[0].n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, bow[1].n_words, dropout_p=0.1).to(device)

enc_opt = optim.SGD(encoder.parameters(), lr=0.01)
dec_opt = optim.SGD(decoder.parameters(), lr=0.01)
loss_fun = nn.NLLLoss()

import random
if debug:
    n_iters = 100
    print_interval = 10
else:
    n_iters = 75000
    print_interval = 100

training_pairs = [tensorsFromPair(random.choice(all_pairs)) for _ in range(n_iters)]
batch_loss = 0

for iteration, pair in enumerate(training_pairs):
    input, label = pair[0], pair[1]

    hidden = encoder.initHidden()
    enc_opt.zero_grad()
    dec_opt.zero_grad()

    input_len = input.size(0)
    label_len = label.size(0)

    codes = torch.zeros(MAX_LENGTH, hidden_size, device=device)

    for i in range(input_len):
        code, hidden = encoder(input[i], hidden)
        codes[i] = code[0, 0]

    decoder_input = torch.tensor([[SOS_idx]], device=device)

    loss = 0

    teacher = True if random.random() < 0.5 else False

    for j in range(label_len):
        decode, hidden, attention = decoder(decoder_input, hidden, codes)
        loss += loss_fun(decode, label[j])
        if not teacher:
            top_val, top_idx = decode.topk(1)
            decoder_input = top_idx.squeeze().detach()
            if decoder_input.item() == EOS_idx:
                break
        else:
            decoder_input = label[j]

    loss.backward()
    enc_opt.step()
    dec_opt.step()

    batch_loss += loss.item() / label_len

    if iteration % print_interval == 0:
        avg_loss = batch_loss / print_interval
        batch_loss = 0
        print('#%u' % iteration, avg_loss)