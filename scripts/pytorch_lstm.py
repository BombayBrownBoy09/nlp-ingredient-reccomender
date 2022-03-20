# import packages
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import re
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
from sklearn.metrics import pairwise
from gensim import corpora
import pickle

# Loading the dataset (i.e. simplified-recipes-1M.npz) and preprocessing to get a list of recipe lists as text
with np.load('data/simplified-recipes-1M.npz', allow_pickle=True) as data:
  recipes = data['recipes']
  ingredients = data['ingredients']
with open('models/PytorchNNEncoder', 'rb') as model_file0:
    NNmodel = pickle.load(model_file0)
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
