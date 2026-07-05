# NSpeech Digit Recognition CNN 

This is a word classification model, trained on the Google Speech Commands dataset. This model uses a CNN to identify numbers spoken from zero to nine with a 99% F1-accuracy rate and 95.9% recognition rate. 

I created this project because I am pursuing ML-audio interaction, and want to create a tool to both help me undestand the inner-workings of this interaction, and create something that could potentionally be useful to others. This project is still a work in progress -- I'm working on expanding its capabilities to recognize live audio. 

## Samples

https://github.com/user-attachments/assets/22aee224-bb40-4b17-bd1d-d22ff3e71e9f

Single word audio clip containing "nine". Model identifies "nine" with confidence 0.9997. 

https://github.com/user-attachments/assets/08221bf5-825a-4a30-83b9-e8ef9e142640

Multi-word audio clip containing "five". Model identifies "five" in the first chunk with confidence 0.9620. In the two suceeding chunks, no number is recognized. 

## Dataset 

The "Google Speech Commands Dataset v2" dataset is used to train model. To identify numbers, 10 digit classes were created ranging from "Zero" to "Nine". 

## Architecture 

### Preprocessing: MFCC Transform

In order to turn audio into something that the CNN can understand, I used an MFCC transform in the preprocessing stage. The MFCC transform takes snippets of the audio clip at a rate of 25 ms with a 10 ms hop length. Following windowing and FFT, 40 mel triangle filters are applied. In doing this, the 1D audio becomes a 2D spectrogram. The choices made here are consistent with typical speech recognition factors for machine learning. [^1]

### Feature Extraction

There are three convolution layers in the CNN. The process of applying these layers allow us to extract distinct features of 2D inputs. The first uses 32 filters to transform the spectrogram into 32 low-level feature maps; the second uses 64 filters to find patterns within the aforementioned feature maps; the third uses 128 filters to turn found patterns into a high-level representation of features. 

Each convolution layer has four steps: convolution, normalization, activation, and pooling. Standard CNN methods are used.

### Transition/Classification/Regularization

Once features are extracted, feature maps are flattened into a 1D vector. This vector is linearly transformed from 4608 channels to 10 channels, to correspondingly fit with labels zero to nine. Between the linear transformations, features are randomly excluded to avoid overfitting. 

## Extract_Digits.py

This program uses the model to find recognized digits from an inputted audio clip. It windows audio clips into 1 second chunks and runs inference.py on each chunk. If a chunk has a confidence levels >= 0.8, then it is labelled as it's corresponding number. Otherwise, it is deemed "unrecognized."  

## Results and Limitations

![Confusion Matrix](results/confusion_matrix.png)

                       precision recall    f1-score  support
        accuracy                           0.99      3937
       macro avg       0.99      0.99      0.99      3937
    weighted avg       0.99      0.99      0.99      3937

Based on Scikit's classification report, the accuracy, macro avg, and weighted avg of the f1-score lie at 99%. Precision and recall, which the f1-score is made of, also lies at 99%. This indicates that this model is exceptional at classifying data that is given to it. The confusion matrix above further supports this argument: errors are seldomly made. 

The model will always output a digit even if the input is not a number. A confidence threshold of 0.8 is used to filter low-confidence predictions, but inputs far outside the training distribution may still produce incorrect high-confidence results.

## How to Run

Check requirements:

    pip install -r requirements.txt

To train model: 
    
    python architecture/training.py

To test: 

    python architecture/extract_digits.py "path/to/audio.wav"

## References

[^1] [Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
