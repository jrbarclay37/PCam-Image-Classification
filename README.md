# PCam-Image-Classification

![pcam](https://github.com/jrbarclay37/PCam-Image-Classification/blob/main/images/pcam.jpeg?raw=true)

## Table of Contents

- [Overview](#overview)
- [Data Exploration](#data-exploration)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Model Deployment](#model-deployment)

## Overview

In this repository, we will be training a Residual Neural Network (ResNet) on image patches of digital pathology scans to detect metastatic cancer. This data is provided by the [PatchCamelyon Grand Challenge](https://patchcamelyon.grand-challenge.org/). As described in the challenge, *"PCam packs the clinically-relevant task of metastasis detection into a straight-forward binary image classification task, akin to CIFAR-10 and MNIST. Models can easily be trained on a single GPU in a couple hours, and achieve competitive scores in the Camelyon16 tasks of tumor detection and whole-slide image diagnosis. Furthermore, the balance between task-difficulty and tractability makes it a prime suspect for fundamental machine learning research on topics as active learning, model uncertainty, and explainability."*

We will be leveraging [Amazon SageMaker](https://aws.amazon.com/sagemaker/) and using their built-in [Image Classification algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html) for this challenge. Our methodology can be divided into the following categories:

**1. Data Exploration** - explore images after loading data into S3 and prepare data for model training.

**2. Hyperparameter Tuning** - run SageMaker hyperparameter tuning job and evaluate results on validation set.

**3. Model Evaluation** - train ResNet model and evaluate results on test set.

**4. Model Deployment** - deploy model for real-time inference.

## Data Exploration

We will be working with the `data-exploration.ipynb` notebook to convert our HDF5 data files into JPEG image files and prepare our data for training in pipe mode by creating augmented manifest files. [Pipe mode](https://aws.amazon.com/blogs/machine-learning/using-pipe-input-mode-for-amazon-sagemaker-algorithms/) allows us to stream our data for training instead of being downloaded first, which results in faster training and reduced disk space utilization.

This notebook will also be used for exploring our images to understand the data we are working with and whether additional data processing may be needed before training our model.


## Hyperparameter Tuning

Now that our data is ready for training, we can launch an [Automatic Model Tuning job](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html) (a.k.a. hyperparameter tuning) in our `hyperparameter-tuning.ipynb` notebook with SageMaker. Unlike classic gridsearch for selecting parameters, automatic model tuning uses Bayesian Search to intelligently choose the best hyperparameters and learn from previous training jobs, significantly reducing the time required to tune your model.

Using the results from our hyperparameter tuning jobs, we discovered that many of our models looked promising, but failed to converge on a solution because the learning rate was too large and overshot. We experimented with reducing the learning rate at certain epochs using the `lr_scheduler` and `lr_factor` parameters to help the model converge. This led to improvement of reaching 88.5% accuracy on our validation set. Our model was still overfitting, so we attempted to solve for this by adjusting the regularization parameters `weight_decay` and `betas`. Unfortunately, this showed only marginal improvement. At this point, we exceeded our budget for hyperparameter tuning and had to move forward.

Additional time and funding would be well spent on further experiments with the learning rate schedules to help the model converge. It is also possible that the quality of the data could be a limitation. We may benefit from applying more sophisticated data augmentation methods for this specific use-case where crop_color_transform may not be the best for histopathology images. [This article](https://towardsdatascience.com/5-ways-to-make-histopathology-image-models-more-robust-to-domain-shifts-323d4d21d889) provides a great analysis on possible data augmentation approaches and the sensitivity of these images to various transformations.

## Model Evaluation

After selecting our parameters from our hyperparameter tuning, we can train our model and evaluate it's performance in the `model-evaluation.ipynb` notebook.

Our final model achieves the following performance:
- **Recall:** 0.876
- **Precision:** 0.880
- **F1 Score:** 0.878
- **AUC:** 0.945
- **Accuracy:** 87.8%

This was data was far more challenging than the popular [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist), so we should be satisfied with our 88% accuracy. However, in the real-world this is not good enough for full automated decision-making, but could be valuable in an AI assistance use-case where this aids the histopathologist by displaying the prediction and confidence.

## Model Deployment

Finally, we can host our model for real-time inference using the `model-deployment.ipynb` notebook.
