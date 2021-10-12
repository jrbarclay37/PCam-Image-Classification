# PCam-Image-Classification

![pcam](https://github.com/jrbarclay37/PCam-Image-Classification/blob/main/images/pcam.jpeg?raw=true)

## Table of Contents

- [Overview](#overview)
- [Data Exploration](#data-exploration)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Model Deployment](#model-deployment)

## Overview

In this repository, I will be training a Residual Neural Network (ResNet) on image patches of digital pathology scans of lymph node sections to detect metastatic cancer. The data provided by the [PatchCamelyon Grand Challenge](https://patchcamelyon.grand-challenge.org/) contains 327,680 color images (96 x 96px) annoted with a binary label indicating presence of metastatic tissue.

As described in the challenge, *"PCam packs the clinically-relevant task of metastasis detection into a straight-forward binary image classification task, akin to CIFAR-10 and MNIST. Models can easily be trained on a single GPU in a couple hours, and achieve competitive scores in the Camelyon16 tasks of tumor detection and whole-slide image diagnosis. Furthermore, the balance between task-difficulty and tractability makes it a prime suspect for fundamental machine learning research on topics as active learning, model uncertainty, and explainability."*

I will be leveraging [Amazon SageMaker](https://aws.amazon.com/sagemaker/) and using their built-in [Image Classification algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html) for this challenge. My methodology can be divided into the following categories:

**1. Data Exploration** - explore images after loading data into S3 and prepare data for model training.

**2. Hyperparameter Tuning** - select hyperparameters with [SageMaker automatic model tuning](https://aws.amazon.com/blogs/aws/sagemaker-automatic-model-tuning/).

**3. Model Evaluation** - train ResNet model and evaluate results on test set.

**4. Model Deployment** - deploy model for real-time inference.

## Data Exploration

I will be working with the `data-exploration.ipynb` notebook to convert the HDF5 data files into JPEG image files and prepare the data for training in pipe mode by creating augmented manifest files. [Pipe mode](https://aws.amazon.com/blogs/machine-learning/using-pipe-input-mode-for-amazon-sagemaker-algorithms/) allows for streaming the data for training instead of being downloaded first, which results in faster training and reduced disk space utilization.

This notebook will also be used for exploring the images to understand the data I am working with and whether additional processing may be needed before training the model.

## Hyperparameter Tuning

Now that the data is ready for training, I will launch an [Automatic Model Tuning job](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html) (a.k.a. hyperparameter tuning) in the `hyperparameter-tuning.ipynb` notebook with SageMaker. Unlike classic gridsearch for selecting parameters, automatic model tuning uses Bayesian Search to intelligently choose the best hyperparameters and learn from previous training jobs, significantly reducing the time required to tune a model.

Using the results from the hyperparameter tuning jobs, I discovered that many of the models looked promising in the first few epochs, but failed to converge on a solution because the learning rate was too large and overshot the local minima. I experimented with reducing the learning rate at certain epochs using the `lr_scheduler` and `lr_factor` parameters to help the model converge. This approach achieved 88.5% accuracy on the validation set. The model was still overfitting, so I attempted to solve for this by adjusting the regularization hyperparameters `weight_decay` and `betas`. Unfortunately, this showed only marginal improvement. At this point, I exceeded my budget for hyperparameter tuning and had to move forward.

Additional time and funding would be well spent on further experiments with the learning rate schedules to help the model converge. It is also possible that the quality of the data could be a limitation. The model may benefit from applying more sophisticated data augmentation methods for this specific use-case where `crop_color_transform` may not be the best for histopathology images. [This article](https://towardsdatascience.com/5-ways-to-make-histopathology-image-models-more-robust-to-domain-shifts-323d4d21d889) provides a great analysis on possible data augmentation approaches and the sensitivity of these images to various transformations.

## Model Evaluation

After selecting the hyperparameters, I train the model and evaluate its performance in the `model-evaluation.ipynb` notebook.

The final model achieves the following performance:
- **Recall:** 0.876
- **Precision:** 0.880
- **F1 Score:** 0.878
- **AUC:** 0.945
- **Accuracy:** 87.8%

This data was noteably more challenging than traditional image classification problems like [MNIST](https://www.tensorflow.org/datasets/catalog/mnist), so I am satisfied with 88% accuracy for this project. However, in the real-world this is insufficient for full automated decision-making, but could be valuable in an AI assistance use-case where the predictions are used to support the histopathologist's diagnosis.

## Model Deployment

Finally, I will host the model for real-time inference by creating a [SageMaker endpoint](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpoint.html) in the `model-deployment.ipynb` notebook.
