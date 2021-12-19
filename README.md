# Brain-Tumor-Detection-Using-Image-Segmentation-
The main purpose of this project was to build a CNN model that would classify if one
has a tumor or not based on the input, i.e MRI scan image using Batch normalization
to train a deep neural network.
In this project, we have described our objective in two parts, the first half deals with
detection of brain tumor that is the presence of the tumor in the provided MRI. The
other part that is the second part contains the classification of the tumor. Here, we will
analyze the MRI images which will conclude the stage of the tumor as benign or
malignant. In general the diagram for our process. The input images will undergo
various stages which can be summarized as follows that are mentioned below:
1. MRI of Brain Images
2. Pre-Processing
3. Feature Extraction
4. Segmentation Technique
5. Image Analysis Figure
Here, We’ve Splitted our dataset into train and test sub-datasets in ratio of 80:20

Tools and Libraries Used
Brain tumor detection project uses the below libraries and frameworks:
• Python – 3.8
• TensorFlow – 2.7
• Keras – 2.7
• Numpy – 1.21.4
• Scikit-learn – 0.20
• Matplotlib – 3.5.1
• Pandas – 1.3.5
• PIL – (Python Image Library)
Data Import and Pre-Processing
Here, We import dataset from the path, and store it in a list in the form of Numpy array.
Pre-processing phase of our project mainly involves those operations that are ordinarily
essential before the goal analysis and extraction of the required data and ordinarily geometric
corrections of the initial image. These enhancements embrace correcting the information for
irregularities and unwanted region noise, removal of non-brain element image and converting
the data so that they are correctly reflected in the original image. The first step of
preprocessing is the conversion of the given input MRI image into a suitable form on which
further work can be performed.


CNN Model



1.1.1 Sequential
A Sequential model is appropriate for a plain stack of layers
where each layer has exactly one input tensor and one output tensor.
Model = Sequential() ##Model creation with no param;
<keras.engine.sequential.Sequential object at 0x000002D99AC42430>
o Sequential groups a linear stack of layers into a tf.keras.Model.
o Sequential provides training and inference features on this model.



1.1.2 Convolution
Convolutional layers are the major building blocks used in convolutional
neural networks.
A convolution is the simple application of a filter to an input that results in an activation.
Repeated application of the same filter to an input results in a map of activations called a
feature map, indicating the locations and strength of a detected feature in an input, such as an
image.
The innovation of convolutional neural networks is the ability to automatically learn a large
number of filters in parallel specific to a training dataset under the constraints of a specific
predictive modeling problem, such as image classification. The result is highly specific
features that can be detected anywhere on input images.
To add the convolution layer, we call the add function with the classifier object and pass in
Convolution2D with parameters. The first argument feature_detectors which is the number of
feature detectors that we want to create. The second and third parameters are dimensions of
the feature detector matrix.
• model.add(Conv2D(32, kernel_size = (2,2), input_shape = (256,256,3), padding =
'Same'))
• model.add(Conv2D(32, kernel_size=(2, 2), activation ='relu', padding = 'Same'))



1.1.3 Batch Normalization
Batch normalization provides an elegant way of reparametrizing almost any
deep network. The reparametrization significantly reduces the problem of coordinating
updates across many layers.
Batch normalization can have a dramatic effect on optimization performance, especially for
convolutional networks and networks with sigmoidal nonlinearities.
Batch normalization can be implemented during training by calculating the mean and
standard deviation of each input variable to a layer per mini-batch and using these statistics to
perform the standardization.
model.add(BatchNormalization())



1.1.4 MaxPooling and Activation ReLU
A pooling layer is another building block of a CNN. It’s function is to progressively reduce
the spatial size of the representation to reduce the amount of parameters and computation in
the network. Pooling layer operates on each feature map independently.
The most common approach used in pooling is max pooling. Max Pooling returns the
maximum value from the portion of the image covered by the Kernel. On the other hand,
Average Pooling returns the average of all the values from the portion of the image covered
by the Kernel. Generally, we use max pooling.
In a neural network, the activation function is responsible for transforming the summed
weighted input from the node into the activation of the node or output for that input.
The rectified linear activation function or ReLU for short is a piecewise linear function that
will output the input directly if it is positive, otherwise, it will output zero. It has become the
default activation function for many types of neural networks because a model that uses it is
easier to train and often achieves better performance.


1.1.5 Dropout and Dens
Deep learning neural networks are likely to quickly overfit a training dataset with few
examples.
## Dropout is a simple way to prevent Neural Networks from Overfitting.


Dropout is a regularization method that approximates training a large number of neural
networks with different architectures in parallel.
During training, some number of layer outputs are randomly ignored or “dropped out.” This
has the effect of making the layer look-like and be treated-like a layer with a different number
of nodes and connectivity to the prior layer. In effect, each update to a layer during training is
performed with a different “view” of the configured layer.
Arguments
• rate: Float between 0 and 1. Fraction of the input units to drop.
• noise_shape: 1D integer tensor representing the shape of the binary dropout mask that
will be multiplied with the input. For instance, if your inputs have shape (batch_size,
timesteps, features) and you want the dropout mask to be the same for all timesteps,
you can use noise_shape=(batch_size, 1, features).
• seed: A Python integer to use as random seed.
• In any neural network, a dense layer is a layer that is deeply connected with its
preceding layer which means the neurons of the layer are connected to every neuron
of its preceding layer. This layer is the most commonly used layer in artificial neural
network networks.
• The dense layer’s neuron in a model receives output from every neuron of its
preceding layer, where neurons of the dense layer perform matrix-vector
multiplication. Matrix vector multiplication is a procedure where the row vector of the
output from the preceding layers is equal to the column vector of the dense layer. The
general rule of matrix-vector multiplication is that the row vector must have as many
columns like the column vector.


1.1.6 Flattening
A flatten layer collapses the spatial dimensions of the input into the channel dimension, all
the pooled feature maps are taken and put into a single vector for inputting it to the next
layer.
The Flatten function flattens all the feature maps into a single long column.
Model.add (Flatten ()) 


1.1.7 Optimizer Adamax
Optimizer that implements the Adamax algorithm.
It is a variant of Adam based on the infinity norm. Default parameters follow those provided
in the paper. Adamax is sometimes superior to adam, specially in models with embeddings.
AdaMax is an extension to the Adam version of gradient descent that generalizes the
approach to the infinite norm (max) and may result in a more effective optimization on some
problems.
AdaMax automatically adapts a separate step size (learning rate) for each parameter in the
optimization problem.
