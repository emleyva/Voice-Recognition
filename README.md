# Number Recognition

This is a word classification model, trained on the Google Speech Commands dataset. This model uses a CNN to identify numbers spoken from zero to nine with a 95.9% accuracy rate. 

## Sample

https://github.com/user-attachments/assets/22aee224-bb40-4b17-bd1d-d22ff3e71e9f

Single word audio clip containing "nine". Uses inference.py. Model identifies "nine" with confidence 0.9997. 

https://github.com/user-attachments/assets/08221bf5-825a-4a30-83b9-e8ef9e142640

Multi-word audio clip containing "five". Uses extract_digits.py. Model identifies "five" in the first chunk with confidence 0.9620. In the two suceeding chunks, no number is recognized. 

## Architecture 

### Preprocessing: MFCC Transform

In order to turn audio into something that the CNN can understand, I used an MFCC transform in the preprocessing stage. The MFCC transform takes snippets of the audio clip at a rate of 25 ms with a 10 ms hop length. Following windowing and FFT, 40 mel triangle filters are applied. In doing this, the 1D audio becomes a 2D spectrogram. The choices made here are consistent with typical speech recognition factors for machine learning. [^1]

### Feature Extraction

There are three convolution layers in the CNN. The process of applying these layers allow us to extract distinct features of 2D inputs. The first uses 32 filters to transform the spectrogram into 32 low-level feature maps; the second uses 64 filters to find patterns within the aforementioned feature maps; the third uses 128 filters to turn found patterns into a high-level representation of features. 

Each convolution layer has four steps: convolution, normalization, activation, and pooling. Standard CNN methods are used.

### Transition/Classification/Regularization

Once features are extracted, feature maps are flattened into a 1D vector. This vector is linearly transformed from 4608 channels to 10 channels, to correspondingly fit with labels zero to nine. Between the linear transformations, features are randomly excluded to avoid overfitting. 


## Dataset 

The "Google Speech Commands Dataset v2" dataset is used to train model. To identify numbers  10 digit classes were created, from "Zero" to "Nine". 

## Results

![Confusion Matrix](results/confusion_matrix.png)

The accuracy, macro avg, and weighted avg of the f1-score lie at 0.99%. This indicates that this model is exceptional at classifying data that is given to it. The confusion matrix above further supports this argument: errors are seldomly made. 

## How to Run

To train model: python architecture/training.py
To test: 
    For single audio files: 
    python architecture/inference.py "path/to/audio.wav"

    For multi-word audio files:
    python architecture/extract_digits.py "path/to/audio.wav"

## Sources

[^1] [Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)


