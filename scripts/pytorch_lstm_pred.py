# import packages
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
from sklearn.metrics import pairwise
from gensim import corpora
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Loading the dataset (i.e. simplified-recipes-1M.npz) and preprocessing to get a list of recipe lists as text
with np.load('data/simplified-recipes-1M.npz', allow_pickle=True) as data:
  recipes = data['recipes']
  ingredients = data['ingredients']
with open('models/PyTorch+LSTM Model', 'rb') as model_file:
    model = pickle.load(model_file)

recipes = [[ingredients[i] for i in recipe] for recipe in recipes]
tokens_list = list(recipes)
vocabulary = []
for sentence in tokens_list:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)
            
# setting a small context to set that near items are part of same recipe
CONTEXT_SIZE = 3
EMBEDDING_DIM = 100
ngrams = [
    (
        [vocabulary[i - j - 1] for j in range(CONTEXT_SIZE)],
        vocabulary[i]
    )
    for i in range(CONTEXT_SIZE, vocabulary_size)
]

vocab = set(vocabulary)
word_to_ix = {word: i for i, word in enumerate(vocab)}

# making n gram language modeler
class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
NNmodel = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(NNmodel.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in ngrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        NNmodel.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = NNmodel(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)

train_len = 3
text_sequences = []
for i in range(train_len,len(vocabulary)):
  seq = vocabulary[i-train_len:i]
  text_sequences.append(seq)
text_sequences 

train_sequence = [[NNmodel.embeddings.weight[word_to_ix[ingredient]].detach().numpy() for ingredient in sequence[:2]] for sequence in text_sequences]
train_sequence

label_sequence = [[NNmodel.embeddings.weight[word_to_ix[ingredient]].detach().numpy() for ingredient in sequence[2:]] for sequence in text_sequences]
label_sequence

print("Enter 2 ingredients:")
input_recipe = input()

# clean input recipe for model predictions
input_text = re.sub(' +', '', input_recipe)
input_text = input_text.lower()
input_text = input_text.split(',')

# converts text to vector
input_vector = [NNmodel.embeddings.weight[word_to_ix[idx]].detach().numpy() if idx in vocabulary else np.zeros((100,)) for idx in input_text]
if len(input_vector)!=2:
  while len(input_vector)!=2:
    input_vector.append(np.zeros((100,)))
input_vector = np.array([input_vector])

#making predictions using model
output_vector = model.predict(input_vector)

# finding the most similar vector to output vector to generate top 10 suggestions
label_sequence = np.array(label_sequence)
reshaped_labels = np.array(label_sequence).reshape([label_sequence.shape[0],label_sequence.shape[2]])
sim_array = sklearn.metrics.pairwise.cosine_similarity(reshaped_labels, Y=output_vector, dense_output=True).reshape([reshaped_labels.shape[0],])
top_10 = np.argsort(sim_array)[-10:][::-1]
print("Top 10 ingredient suggestions:")
for i in top_10:
  print(vocabulary[i], sim_array[i])
