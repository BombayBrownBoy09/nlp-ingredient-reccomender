# nlp-ingredient-reccomender

**Project by Bhargav Shetgaonkar, Nikhil Bhargava and Haoran Cai for Duke AIPI 540 Module 2**
<p align="center"><img align="center" width="800px" src="data/Problem.png"></p>

<a name="proj-stat"></a>
## 1. Problem statement
The objective of this project is to train a deep learning model to recommend complementary ingredients to exist ones using NLP

<a name="proj-struc"></a>
## 2. Project Structure
The project data and codes are arranged in the following manner:

```
├── README.md               <- description of the project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── Makefile [OPTIONAL]     <- set up and run project from the command line
├── main.ipynb]             <- contains the main notebook modeled using Word2Vec and LSTM based approach
├── notebooks               <- contains Pytorch Embeddings + LSTM and Doc2Vec approach
├── .gitignore              <- git ignore file
```

_Data_: <br>
the `data` folder is not a part of this git project as it was heavy. The same can be downloaded from below link:
1) Download data [here](https://github.com/schmidtdominik/RecipeNet/raw/master/simplified-recipes-1M.npz) 
    - **Training data:** Reference Link above you can set a validation split while training
    -  **Validation data:** you can set a 0.15 validation split while training
2) Download trained models [here]()

```sh
https://github.com/schmidtdominik/RecipeNet/raw/master/simplified-recipes-1M.npz
```

<a name="exp"></a>
