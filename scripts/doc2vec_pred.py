# import packages
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import re
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
import pickle

with open('models/doc2vec_model', 'rb') as model_file:
    model = pickle.load(model_file)

print("Enter your current recipe:")
input_recipe = input()

# clean input recipe for model predictions
input_recipe = re.sub(' +', '', input_recipe)
input_recipe = input_recipe.lower()
input_recipe = input_recipe.split(',')

# convert input recipe to d2v vectors 
input_recipe_d2v = model.infer_vector(input_recipe)
input_recipe_d2v = input_recipe_d2v.reshape(-1, len(input_recipe_d2v))
input_recipe_d2v = pd.DataFrame(input_recipe_d2v)

# load ingredient vectors
ingredients_d2v = pd.read_csv('data/outputs/ingredient_doc2vec.csv')
ingredients = ingredients_d2v['ingredient']
ingredients_d2v = ingredients_d2v.drop(columns=['ingredient'])

# get cosine similarioty results
pw_cosine_results = pd.DataFrame(cosine_similarity(input_recipe_d2v, ingredients_d2v))

# get sorted rankings of predictions
predictions = pd.DataFrame()
cs_sim = pw_cosine_results.values[0]
predictions['ingredients'] = ingredients
predictions['cs'] = cs_sim

# print top 10 recommendations
predictions = predictions.sort_values(by='cs', ascending=False).head(10)
t10_preds = list(predictions['ingredients'].values)
print('Have you considered including any of the following?')
for ingredient in t10_preds:
    print('-', ingredient)