# Image Colorization

## Table of contents

* [Overview](#overview)
* [Goals](#goals)
* [Getting Started](#getting-started)
* [Deploy your model](#deploy-your-model)
* [Test our deployment](#test-our-deployment)

## Overview

This repository includes the utilities needed to create and deploy a Convolutional Neural Network that is able to colorize black and white images. This task is being carried out as part of a study in CNNs and should be treated as such.

## Goals

The objective of this project is to train a model that given the l channel of an image to predict its a and b channels and create a demo - proof of consept web app.

## Getting Started

1. Clone repo

```
$ git clone git@github.com:DimitrisPatiniotis/house_valuation.git
```

2. Create a virtual environment and install all requirements listed in requirements.txt

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Deploy your model

To train and deploy the estimator localy you have to follow these steps:

1. Run Processes/main.py and save your model (you can modify the code to give the saved model a name)

```
$ cd Processes
$ python3 main.py
```

2. Deploy your model using the following code

```
$ cd ../
$ python3 -m streamlit run deployment.py
```

Make sure that the model names given in deployment.py are the same you gave in step 1 of this section.

## Test our deployment

You can test our deployment and final model (details can be found in our final report) by visiting www.colo-rize.eu