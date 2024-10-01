# Deep Learning for Deepfake Audio Detection

This project aims to detect Deepfake generated audio by creating three different deep learning models to classify between real and fake audio. The goal is to prevent decietful or malicious AI content by correcting flagging the content as AI generated. 

<br>

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)


## Dataset

The data used comes from DEEP-VOICE: DeepFake Voice Recognition dataset. The dataset is composed of 56 ai audio files, 8 real audio files, and a csv document containing 11779 x 26 features extracted using librosa. Using librosa and the audio files, a new dataset consisting of 10 consecutive seconds datapoints were generated for better recurrent neural network analysis

More information can be found at: [Kaggle Dataset](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition/)


## LSTM Model

The LSTM model consist of 6 layers
<br>
<img src='images/lstm-struct.png' height=400>

## Transformer Model
The Transformer model 
<br>
<img src='images/transformer-struct.png' height=400>

## Wav2Vec2 Model

<br>
<img src='images/wav2vec2-struct.png' height=300>


## Installation

The code requires `python>=3.8`. Required dependencies are found in `requirements.txt`.

Clone the repository locally using

```
git clone https://github.com/kaifanyu/DeepFake-Audio-Detection.git
```

Create a virtual environment for dependencies (optional) then install dependencies using

```
pip install -r requirements.txt
```

## Getting Started

<!-- Model checkpoints can be downloaded at: [https://drive.google.com/drive/folders/1JmsNXJ9Hsimfr2JaOQ9sa2yyxFtptSyK?usp=sharing](https://drive.google.com/drive/folders/sfsdfsdfsdfsdf) -->

The following model checkpoints are available:

- LSTM
- Transformer
- Wav2Vec2

<!-- 
## ML Pipeline

<img src="images/pipeline.png" alt="ML_Pipeline" width="700"/> -->

## Performance

The following shows various aspects for each model.

<div align=center>
<img src="images/result.png" alt="ML_Pipeline" height="300"/>
</div>

<br>

Comparison of LSTM, Transformer v1, and Wav2Vec2 custom model.

Although different datapoints were used between the Wav2Vec2 model as it was extracted using Wav2Vec2 pretrained model, most of the models actually performed relatively the same when scored, with some models just slightly outperforming the others. The Wav2Vec2 gave the best overall accuracy


## Acknowledgements

This project makes use of the base Unet3+ implementation described in the following paper:

[wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/pdf/2006.11477)
