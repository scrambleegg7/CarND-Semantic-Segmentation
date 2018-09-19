# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

[//]: # (Image reference)
[upsample]:./data/samples/upsample.png
[refiner]:./data/samples/refiner.png


[sample1]:./data/samples/um_000084.png
[sample2]:./data/samples/um_000085.png
[sample3]:./data/samples/um_000086.png
[sample4]:./data/samples/um_000087.png
[sample5]:./data/samples/um_000088.png

## Project Object
Project needs to classify roads and non-roads from captured video image frame in the driving lane. Finally, we have to label pixel of roads in the testing image after training CNN architecture. 
As Udacity already provides program framework what kinds of program methods has be prepared at least to build CNN model, we have to fill out python codes fitting implemented 

## Program Structure 
First of all, we copied pre-trained model (vgg16) from central storage area keeping original trained weights, then loads weights using __tf.saved_model.loader.load__ function. In order to match pixel-wise image with ground-truth image, we need to make segmented images from each significant layer (pool3 / pool4 / final layer) combine the integrated pixel images so that would eventually immitate the truth label of images. 

As being displayed in the below black-white image, downsampled __pool3__ now combined with upsampled __2 x pool4__, downsampled pool4 combined with final prediction layer conv7, which meaning of adding skip layer builds model to have local predictions that respect global structur  

![alt text][upsample]


In the final approah making FCN8s, its model builds prediction image very close to Ground truth.

![alt text][refiner]

The below is code structure adding skip layer combining final layer and upsampled layer4.

```
    # Convolutional 1x1 to mantain space information.
    conv_1x1_of_7 = tf.layers.conv2d(vgg_layer7_out,
                                     num_classes,
                                     1, # kernel_size
                                     padding = 'same',
                                     kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                     name='conv_1x1_of_7')

    # Upsample deconvolution x 2
    first_upsamplex2 = tf.layers.conv2d_transpose(conv_1x1_of_7,
                                                  num_classes,
                                                  4, # kernel_size
                                                  strides= (2, 2),
                                                  padding= 'same',
                                                  kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                                  name='first_upsamplex2')

    conv_1x1_of_4 = tf.layers.conv2d(vgg_layer4_out,
                                     num_classes,
                                     1, # kernel_size
                                     padding = 'same',
                                     kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                     name='conv_1x1_of_4')


    # Adding skip layer.
    first_skip = tf.add(first_upsamplex2, conv_1x1_of_4, name='first_skip')

```

## HyperParameter seting up

I have setup following hyperparameters to obtain best result from vgg16 FCN model. Those parameter settings are rational ones to give us rubust training performance. 

```
epochs = 50
batch_size = 4
learning rate = 1e-4
Keep_rate = 0.5 (drop out rate) 
```

## GPU Machine
Instead of leasing cloud service (eg. aws), I have used my own private ubuntu client machine which has installed GTX1080Ti(10GB memory) to train vgg16 model and classify the target road image with CNN architectures.
System log file is saved under root directory while running machine to monitor how loss(cross entropy) is decreased ahead to end of process.
__Finally, I have obtained 0.0147 at 50 epochs.__

```
2018-09-19 20:49:08,613 INFO * average loss : 0.0154  per 72 steps
2018-09-19 20:49:08,615 INFO - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
2018-09-19 20:49:08,615 INFO * Epoch : 50
2018-09-19 20:49:17,985 INFO TRAIN loss: 0.0172  steps:20
2018-09-19 20:49:26,837 INFO TRAIN loss: 0.0112  steps:40
2018-09-19 20:49:35,699 INFO TRAIN loss: 0.0173  steps:60
2018-09-19 20:49:40,855 INFO  -- training --
2018-09-19 20:49:40,856 INFO * average loss : 0.0147  per 72 steps
2018-09-19 20:49:40,856 INFO  finished training ....
Training Finished. Saving test images to: ./runs/1537357780.8563735

```

## Generated Sample images 

Those 5 following images are generated with my training program.

![alt text][sample1]
![alt text][sample2]
![alt text][sample3]
![alt text][sample4]
![alt text][sample5]


### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
