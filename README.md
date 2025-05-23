![alt text](logo.png)


# DeepDiFE

DeepDifE is the core model to classify differential expression of genes under specific stresses, based on cis-regulatory elements. 


## Getting started

Use the `requirements.txt` file to install the necessary python packages using *pip* or *conda*.

## Notebooks
This repository includes several notebooks which can be used as a starting point to experiment with the DeepDiFE analysis toolkit. In this repository, we mainly focus on the prediction of differentially expressed genes in Arabidopsis thaliana under ABA treatment. We divided the complete training and analysis in the following steps

### Training
In the `train_model_tutorial.ipynb`, you will learn how to initialize a DeepDifE model, prepare the training the data and start the training of the CNN model. For this notebook the EvoAug augmentation is enabled, in order to increase the number of training samples. 

If you want to use the PhyloAug augmentation technique, you can find an example in `train_model_phylo_aug.ipynb`. Here orthologous genes from distant species were used to extend the training set. 

Finally, `train_model_cross_validation_tutorial.ipynb` shows how to apply cross-validation training, which is usefull in the case of a low amount of training samples.

### Hyperparameter optimization
The python package *Optuna* was imported to optimize the hyperparameters using the Tree-structured Parzen Estimator algorithm. In the notebook `hyperparameter_optimisation.ipynb` an example can be found where a couple of hyperparameters were tuned.

### Explainability
After training a model with strong predictive performance, the next step is to interpret what sequence patterns drive these predictions.

First, the `saliency_map_tutorial.ipynb` demonstrates how to apply *DeepExplainer* to compute SHAP values based on a selection of input sequences. These can be vizualized in a saliency map highlighting which parts of the sequences contribute to a positive or negative classification.

Next, in the `tfmodisco_tutorial.ipynb` notebook, the selected sequences and their corresponding SHAP values are used to identify and cluster important sequence patterns, known as seqlets. Using a motif database such as JASPAR, these seqlets can be matched to known motifs, enabling the generation of a report that quantifies how existing motifs contribute to the model’s classifications.



