import torch
import torch.nn as nn
import torch.nn.functional as F

import jieba
import nltk
snowball = nltk.stem.SnowballStemmer('english')

def read_lines(file_path):
	fh = open(file_path, encoding='utf-8')
	content = fh.read().strip()
	content = content.lower()
	lines = content.split('\n')
	fh.close()
	return lines

zh_lines = read_lines('dataset/chinese.txt')
en_lines = read_lines('dataset/english.txt')

class BoW:
	def __init__(self):
		self.word2index = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2

	def addWords(self, array):
		for word in array:
			if word not in self.word2index:
				self.word2index[word] = self.n_words
				self.index2word[self.n_words] = word 
				self.n_words += 1

en_bow = BoW()
zh_bow = BoW()

MAX_LENGTH = 30
all_pairs = []

for l, pair in enumerate(zip(en_lines, zh_lines)):
	en_sentence, zh_sentence = pair
	en_words = [snowball.stem(w) for w in en_sentence.split()]
	if len(en_words) > MAX_LENGTH:
		continue
	if l > 300:
		break
	print('line %u' % l, end='\r')
	zh_words = jieba.cut(zh_sentence, cut_all=False)
	# print(en_sentence)
	# print(en_words)
	# print(u" ".join(zh_words))
	en_bow.addWords(en_words)
	zh_bow.addWords(zh_words)
	all_pairs.append(pair)

print('%u pairs of training sentenses' % len(all_pairs))
print('%u total of English words' % en_bow.n_words)
print('%u total of Chinese words' % zh_bow.n_words)
# print(zh_bow.index2word)
# print(en_bow.index2word)
# print(all_pairs)


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
