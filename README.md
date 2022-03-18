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
├── main.ipynb             <- contains the main notebook modeled using Word2Vec and LSTM based approach
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
## 3. Experimentation
We tried 3 approaches here:

**Approach 1(Word2Vec + LSTM):**
The best performing model used Word2vec for vectorization and masking to generate predictors and targets as ingredients or represenattion vectors in our case. We then fine tune vectors using a Neural net with 4 layers (2 LSTM and 2 Dense Layers)
<p align="center"><img align="center" width="800px" src=""></p>

<!-- **Approach 2(With Resnet-152):**
For Deep learning approach we used pretrained Convolutional neural networks. The model is trained with Resnet-152 with no LR scheduler, Resnet-152 with OneCycle LR scheduler and Resnet-152 with Adam. The accuracy for all three models is as shown below
<p align="center"><img align="center" width="800px" src="https://github.com/leocorelli/ComputerVisionProject/blob/main/images/resNetNoLR.png"></p>
<p align="center"><img align="center" width="800px" src="https://github.com/leocorelli/ComputerVisionProject/blob/main/images/resnet1Cycle.png"></p>
<p align="center"><img align="center" width="800px" src="https://github.com/leocorelli/ComputerVisionProject/blob/main/images/resnetAdam.png"></p>

**Approach 3(With Inception V3):**
The model is trained with Incpetion V3. The training accuracy with InceptionV3 is 93.57% and test accuracy is 84.59%
<p align="center"><img align="center" width="800px" src="https://github.com/leocorelli/ComputerVisionProject/blob/main/images/Inceptionv3.png"></p> -->
