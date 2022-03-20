# import packages
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
import random
import pickle

# load data
with np.load('data/raw/simplified-recipes-1M.npz', allow_pickle=True) as data:
    ingredients = data['ingredients']
    recipes = data['recipes']

# load data into list
recipes_list = []
for i in range(len(recipes)):
    try:
        recipes_list.append(list(ingredients[recipes[i]]))
    except:
        # print(i)
        continue

# shuffle array randomly
random.shuffle(recipes_list)

# split into test and train (85/15)
split = round(len(recipes_list)*.85)

# X_train and X_test
X_train = recipes_list[:split]
X_test = recipes_list[split:]

# create doc2vec vocab training
train_corpus = []
for i, sentence in enumerate(X_train):
    train_corpus.append(TaggedDocument(words=sentence, tags=str(i)))

# build and train doc2vec model
model = Doc2Vec(vector_size=50, min_count=2, epochs=25, dm=1)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# output vectors for all ingredients
ingredients_d2v = []
for ingredient in ingredients:
    ingredients_d2v.append(model.infer_vector([ingredient]))
ingredients_d2v = pd.DataFrame(ingredients_d2v)
ingredients_d2v['ingredient'] = ingredients
ingredients_d2v.to_csv('data/outputs/ingredient_doc2vec.csv', index=False)

# save model
pickle.dump(model, open('models/doc2vec_model', 'wb'))