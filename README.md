# PCam-Image-Classification

## Table of Contents

- [Overview](#overview)
- [Data Exploration](#data-exploration)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)

## Overview

In this repository, we will be training a Residual Neural Network (ResNet) on image patches of digital pathology scans to detect metastatic cancer. This data is provided by the [PatchCamelyon Grand Challenge](https://patchcamelyon.grand-challenge.org/). As described in the challenge, "PCam packs the clinically-relevant task of metastasis detection into a straight-forward binary image classification task, akin to CIFAR-10 and MNIST. Models can easily be trained on a single GPU in a couple hours, and achieve competitive scores in the Camelyon16 tasks of tumor detection and whole-slide image diagnosis. Furthermore, the balance between task-difficulty and tractability makes it a prime suspect for fundamental machine learning research on topics as active learning, model uncertainty, and explainability."

We will be leveraging AWS for this challenge and can divide our methodology into the following categories:

**1. Data Exploration** - explore images after loading data into S3 and prepare data for model training.

**2. Hyperparameter Tuning** - run SageMaker hyperparameter tuning job and evaluate results on validation set.

**3. Model Evaluation** - train ResNet model and evaluate results on test set.

**4. Deployment** - deploy model to API.

## Data Exploration

**Reddit**

We will be working with the `scraping_reddit_comments.ipynb` notebook to scrape user comments from r/wallstreetbets. You will need to install python's Reddit API wrapper, `PRAW`.

```console
pip install praw
```

You will also need to [register](https://www.reddit.com/prefs/apps/) an account in order to access the API.

To learn more about the `PRAW` API wrapper, please refer to the [official documentation](https://praw.readthedocs.io/en/latest/).

To mitigate Reddit's slow response times, we also leverage `pushshift.io`. This is a project that warehouses all of Reddit's data, allowing us to query the data more efficiently with significantly faster response times.

To learn more about `pushshift.io`, please refer to the [official documentation](https://pushshift.io/api-parameters/).

**Yahoo Finance**

We will be collecting historical data on TSLA's daily closing prices using `query_tsla_data.ipynb`. You will need to install `yfinance`.

```console
pip install yfinance
```

To learn more about the `yfinance` library, please refer to the [official documentation](https://pypi.org/project/yfinance/).

Additionally, we will use the `TA-Lib` to compute our technical indicators to be used as features in our model. 

```console
pip install TA-Lib
```

To learn more about the `TA-Lib` library, please refer to the [official documentation](https://mrjbq7.github.io/ta-lib/doc_index.html).

## Hyperparameter Tuning

In this section, we will be analyzing our user comments from Reddit and using NLP techniques to engineer scores that measure investor sentiment towards TSLA. This all takes place in the `sentiment_analysis.ipynb` notebook.

We will be relying on the `nltk` library, so you should have this installed.

```console
pip install nltk
```

To learn more about the `nltk` library, please refer to the [official documentation](https://www.nltk.org/).

## Model Evaluation

Before training our machine learning models, we work in the `exploratory_data_analysis.ipynb` notebook to further process and examine the data from the previous sections. This includes feature engineering to enhance the predictive power of variables, as well as force sequence dependancy into the models. 

## Deployment

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
