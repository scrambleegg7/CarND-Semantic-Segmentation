#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


import logging
import time
import sys

import numpy as np
from glob import glob

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_layer = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_layer, keep_prob, layer3, layer4, layer7


    #return None, None, None, None, None
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function


    weights_initializer_stddev = 0.01
    weights_regularized_l2 = 1e-3
    # TODO: Implement function
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
    # Upsample deconvolutions x 2.
    second_upsamplex2 = tf.layers.conv2d_transpose(first_skip,
                                                   num_classes,
                                                   4, # kernel_size
                                                   strides= (2, 2),
                                                   padding= 'same',
                                                   kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                                   name='second_upsamplex2')
    conv_1x1_of_3 = tf.layers.conv2d(vgg_layer3_out,
                                     num_classes,
                                     1, # kernel_size
                                     padding = 'same',
                                     kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                     kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                     name='conv_1x1_of_3')
    # Adding skip layer.
    second_skip = tf.add(second_upsamplex2, conv_1x1_of_3, name='second_skip')
    # Upsample deconvolution x 8.
    third_upsamplex8 = tf.layers.conv2d_transpose(second_skip, num_classes, 16,
                                                  strides= (8, 8),
                                                  padding= 'same',
                                                  kernel_initializer = tf.random_normal_initializer(stddev=weights_initializer_stddev),
                                                  kernel_regularizer= tf.contrib.layers.l2_regularizer(weights_regularized_l2),
                                                  name='third_upsamplex8')
    return third_upsamplex8


    #return None
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logging.info(" -- Optimizer --")

    with tf.name_scope('Loss'):

        logits = tf.reshape(nn_last_layer, (-1, num_classes))
        correct_label_flat = tf.reshape(correct_label, (-1,num_classes))

        # optionality - create accuracy tag
        pred_up = tf.argmax(nn_last_layer, axis=3)
        correct_label_up = tf.argmax(correct_label,axis=3)        


    with tf.name_scope('Optimizer'):

        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label_flat)
        cross_entropy_loss = tf.reduce_mean(unweighted_losses)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.01  # Choose an appropriate one.
        loss = cross_entropy_loss + reg_constant * sum(reg_losses)    
        
        # Define optimizer. Adam in this case to have variable learning rate.
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        # Apply optimizer to the loss function.
        train_op = optimizer.minimize(loss) # including regularize loss    
        

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate


    """
    # TODO: Implement function

    # save the trained model or read trained model from directory

    logging.info("- " * 40)
    
    """
    saver = tf.train.Saver()
    model_filename = "./save_models/vgg_semantic"
    ckpt = tf.train.get_checkpoint_state('./save_models/')
    if ckpt: # if checkpoint found
        last_model = ckpt.model_checkpoint_path # path for last saved model

        load_step = int(os.path.basename(last_model).split('-')[1])
        load_step += 1
        
        logging.info("load model file.." + last_model + str( load_step ))
        saver.restore(sess, last_model) # restore all parameters

        init_local = tf.local_variables_initializer()
        sess.run( [init_local])

    else: # if model NOT found 
        logging.info("- " * 40)
        logging.info(" model NOT found, all variables are initialized.")

        # initialize for global and local    
        init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        sess.run( [init, init_local])

        load_step = 0

    epochs = epochs + load_step
    """

    logging.info("- " * 40)
    logging.info(" model NOT found, all variables are initialized.")

    # initialize for global and local    
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run( [init, init_local])



    for epoch in range(epochs):

        logging.info("- " * 40)
        logging.info('* Epoch : {}'.format(epoch + 1))
        loss_log = []

        #
        # read training image and grand truth label
        #
        for idx, ( image, label) in enumerate( get_batches_fn(batch_size) ):
            
            feed_dict={ 
                input_image: image, 
                correct_label: label, 
                keep_prob: 0.5,
                learning_rate : 1e-6     # 1e-4
                }

            _, loss = sess.run( [train_op, cross_entropy_loss], feed_dict=feed_dict  )

            if idx % 20 == 0 and idx != 0:
                logging.info("TRAIN loss: %.4f  steps:%d" % ( loss, idx )  )
                
            loss_log.append(loss)

        logging.info(" -- training --")
        logging.info("* average loss : %.4f  per %d steps" % ( np.mean( loss_log  ) , idx )   )
    
    logging.info(" finished training ....")
    #logging.info(" model saved..")
    #saver.save(sess, model_filename, global_step=epoch)


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    epochs = 50
    batch_size = 4

    data_folder = os.path.join(data_dir, 'data_road/training')
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    logging.info(" image counters : %d" % len(image_paths))
    logging.info(" batch sizes : %d" % batch_size )

    training_steps_per_epch = int( len(image_paths) / batch_size )
    logging.info("training steps per epoch: : %d " % training_steps_per_epch )

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # Layer_outputs = logits
        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg( sess, vgg_path )
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # TODO: Train NN using the train_nn function

        # setup placeholder for GrandTrue label and learning rate 
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        

        # call optimizer 
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        #
        # call training process with several key parameters
        # 
        #   batch_size : 2-5
        #   get_batches_fn : generator function
        
        #   < from Optimizer > 
        #   train_op : training operation
        #   cross_entropy_loss : cross entropy loss
        #   
        #   <image and grand truth>
        #   input_image : shrinked to 160x576
        #   keep_prob : probability for dropout from loading vgg model
        #   learning_rate : 


        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate )



        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
