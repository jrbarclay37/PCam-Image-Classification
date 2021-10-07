# PCam-Image-Classification

## UNDER CONSTRUCTION
__________________________________________________________________

## Table of Contents

- [Overview](#overview)
- [Data Exploration](#data-exploration)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Model Deployment](#model-deployment)

## Overview

In this repository, we will be training a Residual Neural Network (ResNet) on image patches of digital pathology scans to detect metastatic cancer. This data is provided by the [PatchCamelyon Grand Challenge](https://patchcamelyon.grand-challenge.org/). As described in the challenge, *"PCam packs the clinically-relevant task of metastasis detection into a straight-forward binary image classification task, akin to CIFAR-10 and MNIST. Models can easily be trained on a single GPU in a couple hours, and achieve competitive scores in the Camelyon16 tasks of tumor detection and whole-slide image diagnosis. Furthermore, the balance between task-difficulty and tractability makes it a prime suspect for fundamental machine learning research on topics as active learning, model uncertainty, and explainability."*

We will be leveraging Amazon SageMaker and using their built-in [Image Classification algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html) for this challenge. Our methodology can be divided into the following categories:

**1. Data Exploration** - explore images after loading data into S3 and prepare data for model training.

**2. Hyperparameter Tuning** - run SageMaker hyperparameter tuning job and evaluate results on validation set.

**3. Model Evaluation** - train ResNet model and evaluate results on test set.

**4. Deployment** - deploy model to API.

## Data Exploration

We will be working with the `data-exploration.ipynb` notebook to convert our HDF5 data files into JPEG image files and prepare our data for training in pipe mode by creating augmented manifest files. [Pipe mode](https://aws.amazon.com/blogs/machine-learning/using-pipe-input-mode-for-amazon-sagemaker-algorithms/) allows us to stream our data for training instead of being downloaded first, which results in faster training and reduced disk space utilization.

This notebook will also be used for exploring our images to understand the data we are working with and whether additional data processing may be needed before training our model.


## Hyperparameter Tuning



In this section, we will be analyzing our user comments from Reddit and using NLP techniques to engineer scores that measure investor sentiment towards TSLA. This all takes place in the `sentiment_analysis.ipynb` notebook.

We will be relying on the `nltk` library, so you should have this installed.

```console
pip install nltk
```

To learn more about the `nltk` library, please refer to the [official documentation](https://www.nltk.org/).

## Model Evaluation

After selecting our parameters from our hyperparameter tuning, we can train our model and evaluate it's performance in the `model-evaluation.ipynb` notebook.



Our final model achieves the following performance:
- Recall: 0.88
- Precision: 0.88
- F1 Score: 0.88
- AUC: 0.945
- Accuracy: 88%

Our model was still

This was data was far more challenging than the popular [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist), so we should be satisfied with our 88% accuracy. However, 

## Model Deployment

Bringing everything together, we use our sentiment scores and technical indicators to predict the future price of TSLA. We use a simple ARIMA model as our baseline in the `arima_forecasting` notebook, and then attempt to improve performance using the following models in our `ml_forecasting.ipynb` notebook:
- Random Forest
- XGBoost
- LSTM

For these final workbooks, you should have `statsmodels`, `scikit-learn`, `xgboost`, and `keras` installed on your machine. 

```console
pip install statsmodels
```

```console
pip install scikit-learn
```

```console
pip install xgboost
```

```console
pip install keras
```

To learn more about the documentation for each of these libraries, please refer to the following links:
- [statsmodels](https://www.statsmodels.org/stable/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)
- [xgboost](https://xgboost.readthedocs.io/en/latest/)
- [keras](https://keras.io/about/)
