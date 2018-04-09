Initial commit contains the neural network required for the character recoginition....

This uses a typical 3-layer multi-layer perceptron. Training time is about 3 seconds for 80% correct, and 30s for 96% correct against the competition MNIST set of hand-written digits, which is pretty speedy for a simple C program.

  1)Hand-written linear algebra code in C, refined with asm inspection, and cache efficiency. e.g. replacement for numpy etc.        Matrix multiplication etc.
  2)Hand-written CSV parser for reading the data.
  3)Hand-written image output for testing the data.
  
Note that I didn't include the test data files because they are pretty big. I am including the links for training and testing..


Traing set images:
http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

Training set labels:
http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

Test set images:
http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz

Test set labels:
http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
