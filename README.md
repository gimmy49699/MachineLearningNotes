# MachineLearningNotes
My study&amp;review notes on machine learning.

code enviroment:

Python 3.6.5 | Tensorflow 1.13.1 | Pytorch 1.2.0 | Sci-kit 0.22.1 | Keras 2.2.4

20-02-29 updates:
  - Logistic Regression Algorithm & Implementing with numpy.
  <img src="https://github.com/gimmy49699/MachineLearningNotes/blob/master/MachineLearningCode/pictures/LR.jpg" height="300" div align=center />

20-03-02 updates:
  - Linear Discriminant Analysis & Implementing with numpy.
  <img src="https://github.com/gimmy49699/MachineLearningNotes/blob/master/MachineLearningCode/pictures/wl3aDataDistribution.png" height="300" div align=center />
  <img src="https://github.com/gimmy49699/MachineLearningNotes/blob/master/MachineLearningCode/pictures/LDA.jpg" height="300" div align=center />
  
  - Principal Component Analysis & Implementing with numpy.
      - Both PCA and LDA are methods of reducing feature dimensions.
      - LDA is a supervised method while PCA is unsupervised.
      - LDA can be used as classification method.
      - PCA cares about the principal features of datas while LDA cares about seprating each categories.
      - Both eigenvalue decomposition and singluar value decomposition can be used in PCA or LDA.
      - Better centeralizing the datas while using PCA.
    <img src="https://github.com/gimmy49699/MachineLearningNotes/blob/master/MachineLearningCode/pictures/PCA.jpg" height="300" div align=center />

20-03-04 updates:
  - Decision Tree & Implementing with numpy.
      - Implemented ID3.
      - Information entropy, conditional entropy, information gain, information gain ratio.
      - Recursively building decision tree and pruning.
    <img src="https://github.com/gimmy49699/MachineLearningNotes/blob/master/MachineLearningCode/pictures/decisiontree.png" div align=center />

20-03-05 updates:
  - Neural Network & Implementing with numpy.
    - Implemented basic fully connected neural network with numpy.
    - Sigmoid activation function only, updates in subsequent version.
      - newly updated model architecture: tanh -> tanh -> ... -> sigmoid
    - Basic back propagation algorithm only, updates in subsequent version.
    - Choose hidden layers, hidden units, epochs and batch size artificially before start training.
  <img src="https://github.com/gimmy49699/MachineLearningNotes/blob/master/MachineLearningCode/pictures/NN.jpg" height="300" div align=center>
