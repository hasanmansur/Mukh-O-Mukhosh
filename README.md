# Mukh-O-Mukhosh (Bengali: মুখ ও মুখোশ)

- A Convolutional Neural Network model for Face Mask Detection
- The name Mukh-O-Mukhosh is inspired from the name of the the first Bengali language feature film 'Mukh O Mukhosh' (Bengali: মুখ ও মুখোশ, lit. 'The Face and the Mask') which was released back in 1956. https://en.wikipedia.org/wiki/Mukh_O_Mukhosh
- This model will be trained on the dataset created by [prajnasb] [https://github.com/prajnasb/observations]

## Roadmap:

- Background Study:
  - [x] Artificial Neural Network (ANN) basics: perceptron, neural network, activation function, cost function, gradient descent, back propagation
  - [x] Simple learning: 
    - [x] neural network with single input & single output feature
    - [x] example: linear regression 
  - [x] Basic architecture of ANN: 
    - [x] multiple input features , hidden layers & multiple output features
    - [x] data loading & splitting into train/test set
    - [x] example: multiclass classification from continuous data (IRIS dataset)
  - [x] General architecture of ANN:
    - [x] concept of feature engineering, continuous & categorical data, embedding, batch normalization, dropout layer
    - [x] example: regression from a mix of continuous & categorical data (NYC taxi dataset)
    - [x] example: multiclass classification from a mix of continuous & categorical data (NYC taxi dataset)
  - [x] Convolutional Neural Network:
    - [x] getting familiarized with a basic image dataset (MNIST)
    - [x] example: MNIST with ANN
    - [x] motvation behind choosing CNN over ANN
    - [x] basic concepts of CNN
    - [x] example: MNIST with CNN
  
- Implementation:
    - [x] dataset preparation
      - [x] read images from directory
      - [x] data preprocessing (resize, toTensor etc)
    - [x] data loading
      - [x] creating data loader (attr: batch size, shuffle/randomize etc)
    - [x] defining model, loss function & optimizer
    - [x] train the model
    - [x] plot loss & accuracy (for training & validation)
    - [x] experimenting with different hyperparams
    - [x] training & validating using gpu/cuda

<!-- ## Evolution of Deep Neural Networks(DNN)
core components/concepts -> building blocks of DNN -> major DNN architectures -> family tree of major DNN architectures

### core components/concepts of DNN
  - Parameters
  - Layers
  - Activation functions
  - Loss functions
  - Optimization methods
  - Hyperparameters

### building block networks of DNN
  - Feed-forward multilayer neural networks
  - RBMs
  - Autoencoders

### major DNN architectures
  - Unsupervised Pretrained Networks (UPNs)
  - Convolutional Neural Networks (CNNs)
  - Recurrent Neural Networks
  - Recursive Neural Networks
  
### family tree of major DNN architectures
 -->
