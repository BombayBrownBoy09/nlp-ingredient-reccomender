# import packages
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import re
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
import pickle

# Loading the dataset (i.e. simplified-recipes-1M.npz) and preprocessing to get a list of recipe lists as text
with np.load('data/simplified-recipes-1M.npz', allow_pickle=True) as data:
  recipes = data['recipes']
  ingredients = data['ingredients']

recipes = [[ingredients[i] for i in recipe] for recipe in recipes]
word2vec = Word2Vec(recipes, min_count=1)

with open('models/Word2Vec + LSTM Model', 'rb') as model_file:
    model = pickle.load(model_file)

print("Enter your current recipe:")
input_recipe = input()

# clean input recipe for model predictions
input_text = re.sub(' +', '', input_recipe)
input_text = input_text.lower()
input_text = input_text.split(',')

# converts text to vector
input_vector = [word2vec[idx] if idx in word2vec else np.zeros((100,)) for idx in input_text]
if len(input_vector)!=2:
  while len(input_vector)!=2:
    input_vector.append(np.zeros((100,)))

# converts input vector to numpy array
input_vector = np.array([input_vector])

# getting model predictions for top 10 suggestions by finding the 10 most similar ingredients to our output
output_vector = model.predict(input_vector)
pred = word2vec.most_similar(positive=[output_vector.reshape(100,)], topn=10)
print("Top 10 ingredient suggestions:")
pred