from math import ceil
import numpy as np

import theano
import theano.tensor as t


###############################################################
# Functions used by layers but do not change Layer Attributes #
###############################################################


def apply_dropout(rng, dropout_rate, input_train_shape, input_train, input_inference, input_testing):
    if dropout_rate > 0.001:  # Below 0.001 I take it as if there is no dropout at all. (To avoid float problems with
        # == 0.0. Although my tries show it actually works fine.)
        activation_probability = (1 - dropout_rate)
        # TODO: possible bug next two lines with dropout, tried numpy version of random state
        random_stream = np.random.RandomState(rng.randint(999999))
        dropout_mask = random_stream.binomial(n=1, p=activation_probability, size=input_train_shape)
        dropout_input_image = input_train * dropout_mask
        dropout_inference_input_image = input_inference * activation_probability
        dropout_testing_input_image = input_testing * activation_probability
    else:
        dropout_input_image = input_train
        dropout_inference_input_image = input_inference
        dropout_testing_input_image = input_testing
    return dropout_input_image, dropout_inference_input_image, dropout_testing_input_image


def apply_batch_norm(batch_norm_rolling_average, input_train, input_val, input_test, input_train_shape):
    """
    :param batch_norm_rolling_average: the rolling average for batch normalization over that many batches
    :param input_train: the training input
    :param input_val: the input value
    :param input_test: the testing input
    :param input_train_shape: the shape of the training input
    :return:
    """

    num_channels = input_train_shape[1]
    g_batch_norm_values = np.ones(num_channels, dtype='float32')
    g_batch_norm = theano.shared(value=g_batch_norm_values, borrow=True)
    b_batch_norm_values = np.zeros(num_channels, dtype='float32')
    b_batch_norm = theano.shared(value=b_batch_norm_values, borrow=True)

    # for rolling average:
    batch_norm_zeros = np.zeros((batch_norm_rolling_average, num_channels), dtype='float32')
    mu_bns_array_for_rolling_average = theano.shared(batch_norm_zeros, borrow=True)

    batch_norm_ones = np.ones((batch_norm_rolling_average, num_channels), dtype='float32')
    var_bns_array_for_rolling_average = theano.shared(batch_norm_ones, borrow=True)

    shared_new_mu_b = theano.shared(np.zeros(num_channels, dtype='float32'), borrow=True)
    shared_new_var_b = theano.shared(np.ones(num_channels, dtype='float32'), borrow=True)

    e1 = np.finfo(np.float32).tiny
    # WARN, PROBLEM, THEANO BUG. The below was returning (True,) instead of a vector, if I have only 1 FM. (Vector is
    #  (False,)). Think I corrected this bug.
    # average over all axis but the 2nd, which is the FM axis.
    mu_b = input_train.mean(axis=[0, 2, 3, 4])
    # The above was returning a broadcast-able (True,) tensor when FM-number=1. Here I make it a broadcast-able (False),
    #  which is the "vector" type. This is the same type with the shared_new_mu_b, which we are updating with this.
    #  They need to be of the same type.
    mu_b = t.unbroadcast(mu_b, 0)
    var_b = input_train.var(axis=[0, 2, 3, 4])
    var_b = t.unbroadcast(var_b, 0)
    var_b_plus_e = var_b + e1

    # ---computing mu and var for inference from rolling average---
    mu_rolling_average = mu_bns_array_for_rolling_average.mean(axis=0)
    # batchSize*voxels in a featureMap. See p5 of the paper.
    effective_size = input_train_shape[0] * input_train_shape[2] * input_train_shape[3] * input_train_shape[4]
    var_rolling_average = (effective_size / (effective_size - 1)) * var_bns_array_for_rolling_average.mean(axis=0)
    var_rolling_average_plus_e = var_rolling_average + e1

    # OUTPUT FOR TRAINING
    norm_yi_train = _get_training_output(b_batch_norm, g_batch_norm, input_train, mu_b, var_b_plus_e)

    # OUTPUT FOR VALIDATION
    dimshuffle_sqrt, norm_yi_val = \
        _get_validation_output(b_batch_norm, g_batch_norm, input_val, mu_rolling_average, var_rolling_average_plus_e)

    # OUTPUT FOR TESTING
    norm_yi_test = _get_testing_output(b_batch_norm, dimshuffle_sqrt, g_batch_norm, input_test, mu_rolling_average)

    # var_b is the current value of muB calculated in this training iteration. It will be saved in the
    # "shared_new_mu_b" (update), in order to be used for updating the rolling average. Something could be
    # simplified here.
    return (norm_yi_train, norm_yi_val, norm_yi_test, g_batch_norm, b_batch_norm, mu_bns_array_for_rolling_average,
            var_bns_array_for_rolling_average, shared_new_mu_b, shared_new_var_b, mu_b, var_b)


def _get_training_output(b_batch_norm, g_batch_norm, input_train, mu_b, var_b_plus_e):
    sqrt_dimshuffle = t.sqrt(var_b_plus_e.dimshuffle('x', 0, 'x', 'x', 'x'))
    norm_xi_train = (input_train - mu_b.dimshuffle('x', 0, 'x', 'x', 'x')) / sqrt_dimshuffle
    # dimshuffle makes b broadcast-able.
    batch_norm_xi_dimshuffle = g_batch_norm.dimshuffle('x', 0, 'x', 'x', 'x') * norm_xi_train
    norm_yi_train = batch_norm_xi_dimshuffle + b_batch_norm.dimshuffle('x', 0, 'x', 'x', 'x')
    return norm_yi_train


def _get_testing_output(b_batch_norm, dimshuffle_sqrt, g_batch_norm, input_test, mu_rolling_average):
    norm_xi_test = (input_test - mu_rolling_average.dimshuffle('x', 0, 'x', 'x', 'x')) / dimshuffle_sqrt
    norm_xi_test_dimshuffle = g_batch_norm.dimshuffle('x', 0, 'x', 'x', 'x') * norm_xi_test
    norm_yi_test = norm_xi_test_dimshuffle + b_batch_norm.dimshuffle('x', 0, 'x', 'x', 'x')
    return norm_yi_test


def _get_validation_output(b_batch_norm, g_batch_norm, input_val, mu_rolling_average, var_rolling_average_plus_e):
    dimshuffle_sqrt = t.sqrt(var_rolling_average_plus_e.dimshuffle('x', 0, 'x', 'x', 'x'))
    norm_xi_val = (input_val - mu_rolling_average.dimshuffle('x', 0, 'x', 'x', 'x')) / dimshuffle_sqrt
    xi_val_dimshuffle = g_batch_norm.dimshuffle('x', 0, 'x', 'x', 'x') * norm_xi_val
    norm_yi_val = xi_val_dimshuffle + b_batch_norm.dimshuffle('x', 0, 'x', 'x', 'x')
    return dimshuffle_sqrt, norm_yi_val


def apply_bias_to_fms(fms_train, fms_val, fms_test, num_fms):
    b_values = np.zeros(num_fms, dtype='float32')
    b = theano.shared(value=b_values, borrow=True)
    fms_with_bias_applied_train = fms_train + b.dimshuffle('x', 0, 'x', 'x', 'x')
    fms_with_bias_applied_val = fms_val + b.dimshuffle('x', 0, 'x', 'x', 'x')
    fms_with_bias_applied_test = fms_test + b.dimshuffle('x', 0, 'x', 'x', 'x')
    return b, fms_with_bias_applied_train, fms_with_bias_applied_val, fms_with_bias_applied_test


def apply_relu(input_train, input_val, input_test):
    # input is a tensor of shape (batchSize, FMs, r, c, z)
    output_train = t.maximum(0, input_train)
    output_val = t.maximum(0, input_val)
    output_test = t.maximum(0, input_test)
    return output_train, output_val, output_test


def apply_prelu(input_train, input_val, input_test, num_input_channels):
    # input is a tensor of shape (batchSize, FMs, r, c, z)
    # "Delving deep into rectifiers" initializes it like this. LeakyRelus are at 0.01
    a_prelu_values = np.ones(num_input_channels, dtype='float32') * 0.01
    a_prelu = theano.shared(value=a_prelu_values, borrow=True)  # One separate a (activation) per feature map.
    broad_casted_prelu_with_channels = a_prelu.dimshuffle('x', 0, 'x', 'x', 'x')

    pos_train = t.maximum(0, input_train)
    neg_train = broad_casted_prelu_with_channels * (input_train - abs(input_train)) * 0.5
    output_train = pos_train + neg_train
    pos_val = t.maximum(0, input_val)
    neg_val = broad_casted_prelu_with_channels * (input_val - abs(input_val)) * 0.5
    output_val = pos_val + neg_val
    pos_test = t.maximum(0, input_test)
    neg_test = broad_casted_prelu_with_channels * (input_test - abs(input_test)) * 0.5
    output_test = pos_test + neg_test

    return a_prelu, output_train, output_val, output_test


def apply_elu(alpha, input_train, input_val, input_test):
    output_train = t.basic.switch(input_train > 0, input_train, alpha * t.basic.expm1(input_train))
    output_val = t.basic.switch(input_val > 0, input_val, alpha * t.basic.expm1(input_val))
    output_test = t.basic.switch(input_test > 0, input_test, alpha * t.basic.expm1(input_test))
    return output_train, output_val, output_test


def apply_selu(input_train, input_val, input_test):
    # input is a tensor of shape (batchSize, FMs, r, c, z)
    lambda01 = 1.0507  # calc in p4 of paper.
    alpha01 = 1.6733

    output_train, output_val, output_test = apply_elu(alpha01, input_train, input_val, input_test)
    output_train = lambda01 * output_train
    output_val = lambda01 * output_val
    output_test = lambda01 * output_test

    return output_train, output_val, output_test


def init_weight_tensor(filter_shape, conv_weight_init_method, rng):
    # create an initialize the weight tensors
    # filter_shape of dimensions: [#FMs in this layer, #FMs in input, rKernelDim, cKernelDim, zKernelDim]
    std_for_init = None
    if conv_weight_init_method[0] == "normal":
        std_for_init = conv_weight_init_method[1]  # commonly 0.01 from Krizhevski
    elif conv_weight_init_method[0] == "fanIn":
        variance_scale = conv_weight_init_method[1]  # 2 for init ala Delving into Rectifier, 1 for SNN.
        std_for_init = np.sqrt(variance_scale / (filter_shape[1] * filter_shape[2] * filter_shape[3] * filter_shape[4]))

    # Perhaps I want to use: theano.config.floatX in the below
    filter_shape = (filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3], filter_shape[4])
    w_init_np_array = np.asarray(rng.normal(loc=0.0, scale=std_for_init, size=filter_shape), dtype='float32')
    weight = theano.shared(w_init_np_array, borrow=True)
    # weight shape: [#FMs of this layer, #FMs of Input, rKernFims, cKernDims, zKernDims]
    return weight


def convolve_weight_matrix(w, filter_shape, conv_train_input, conv_val_input, conv_test_input,
                           conv_train_input_shape, conv_val_input_shape, conv_test_input_shape):
    # input weight matrix W has shape: [ #ChannelsOut, #ChannelsIn, R, C, Z ] == filterShape
    # filterShape is the shape of W.
    # Input signal given in shape [BatchSize, Channels, R, C, Z]

    conv_train_output, w_reshaped, w_reshaped_shape = convolve_train(conv_train_input, conv_train_input_shape,
                                                                     filter_shape, w)
    conv_val_output = convolve_validation(conv_val_input, conv_val_input_shape, w_reshaped, w_reshaped_shape)
    conv_test_output = convolve_testing(conv_test_input, conv_test_input_shape, w_reshaped, w_reshaped_shape)

    # reshape result, back to the shape of the input image
    output_train = conv_train_output.dimshuffle(0, 1, 3, 4, 2)
    output_val = conv_val_output.dimshuffle(0, 1, 3, 4, 2)
    output_test = conv_test_output.dimshuffle(0, 1, 3, 4, 2)

    output_shape_train = [conv_train_input_shape[0], filter_shape[0],
                          conv_train_input_shape[2] - filter_shape[2] + 1,
                          conv_train_input_shape[3] - filter_shape[3] + 1,
                          conv_train_input_shape[4] - filter_shape[4] + 1]

    output_shape_val = [conv_val_input_shape[0], filter_shape[0],
                        conv_val_input_shape[2] - filter_shape[2] + 1,
                        conv_val_input_shape[3] - filter_shape[3] + 1,
                        conv_val_input_shape[4] - filter_shape[4] + 1]

    output_shape_test = [conv_test_input_shape[0], filter_shape[0],
                         conv_test_input_shape[2] - filter_shape[2] + 1,
                         conv_test_input_shape[3] - filter_shape[3] + 1,
                         conv_test_input_shape[4] - filter_shape[4] + 1]

    return output_train, output_val, output_test, output_shape_train, output_shape_val, output_shape_test


def convolve_testing(conv_test_input, conv_test_input_shape, w_reshaped, w_reshaped_shape):
    conv_reshaped_test_input = conv_test_input.dimshuffle(0, 1, 4, 2, 3)
    conv_reshaped_shape_test_input = (conv_test_input_shape[0], conv_test_input_shape[1], conv_test_input_shape[4],
                                      conv_test_input_shape[2], conv_test_input_shape[3])

    conv_test_output = t.nnet.conv3d(input=conv_reshaped_test_input,
                                     filters=w_reshaped,
                                     input_shape=conv_reshaped_shape_test_input,
                                     filter_shape=w_reshaped_shape,
                                     border_mode='valid',
                                     subsample=(1, 1, 1),
                                     filter_dilation=(1, 1, 1))
    return conv_test_output


def convolve_validation(conv_val_input, conv_val_input_shape, w_reshaped, w_reshaped_shape):
    conv_reshaped_val_input = conv_val_input.dimshuffle(0, 1, 4, 2, 3)
    conv_reshaped_shape_val_input = (conv_val_input_shape[0], conv_val_input_shape[1], conv_val_input_shape[4],
                                     conv_val_input_shape[2], conv_val_input_shape[3])
    conv_val_output = t.nnet.conv3d(input=conv_reshaped_val_input,
                                    filters=w_reshaped,
                                    input_shape=conv_reshaped_shape_val_input,
                                    filter_shape=w_reshaped_shape,
                                    border_mode='valid',
                                    subsample=(1, 1, 1),
                                    filter_dilation=(1, 1, 1))
    return conv_val_output


def convolve_train(conv_train_input, conv_train_input_shape, filter_shape, w):
    # Conv3d requires filter shape: [ #ChannelsOut, #ChannelsIn, Z, R, C ]
    w_reshaped = w.dimshuffle(0, 1, 4, 2, 3)
    w_reshaped_shape = (filter_shape[0], filter_shape[1], filter_shape[4], filter_shape[2], filter_shape[3])
    # Conv3d requires signal in shape: [BatchSize, Channels, Z, R, C]
    reshaped_train_conv_input = conv_train_input.dimshuffle(0, 1, 4, 2, 3)
    reshaped_shape_train_conv_input = (conv_train_input_shape[0], conv_train_input_shape[1],
                                       conv_train_input_shape[4], conv_train_input_shape[2], conv_train_input_shape[3])
    # TODO: investigate this, maybe we can swap with TF?
    # Output is in the shape of the input image (signals_shape).
    conv_train_output = t.nnet.conv3d(input=reshaped_train_conv_input,
                                      # batch_size, time, num_of_input_channels, rows, columns
                                      filters=w_reshaped,
                                      # Number_of_output_filters, Z, Numb_of_input_Channels, r, c
                                      input_shape=reshaped_shape_train_conv_input,  # Can be None. Used for optimization
                                      filter_shape=w_reshaped_shape,  # Can be None. Used for optimization.
                                      border_mode='valid',
                                      subsample=(1, 1, 1),  # strides
                                      filter_dilation=(1, 1, 1))  # dilation rate
    return conv_train_output, w_reshaped, w_reshaped_shape


def apply_softmax(softmax_input, softmax_shape_input, num_output_classes, softmax_temperature):
    # Apply softmax to feature maps and return the y probability and y prediction The softmax function works on 2D
    # tensors (matrices). It computes the softmax for each row. Rows are independent, eg different samples in the
    # batch. Columns are the input features, eg class-scores. Softmax's input 2D matrix should have shape like:
    # [data_samples, #Classess ] My class-scores/class-FMs are a 5D tensor (batchSize, #Classes, r, c, z). I need to
    # reshape it to a 2D tensor. The reshaped 2D Tensor will have dimensions: [ batchSize * r * c * z , #Classses ]
    # The order of the elements in the rows after the reshape should be:
    softmax_reshaped_input = softmax_input.dimshuffle(0, 2, 3, 4, 1)
    softmax_flattened_input = softmax_reshaped_input.flatten(1)

    # flatten is "Row-major" 'C' style. ie, starts from index [0,0,0] and grabs elements in order such that last dim
    # index increases first and first index increases last. (first row flattened, then second follows, etc)
    num_densely_classified_voxels = softmax_shape_input[2] * softmax_shape_input[3] * softmax_shape_input[4]
    softmax2d_first_dimension = softmax_shape_input[0] * num_densely_classified_voxels  # batchSize*r*c*z.

    # Reshape works in "Row-major", ie 'C' style too.
    softmax2d_input = softmax_flattened_input.reshape((softmax2d_first_dimension, num_output_classes))

    # Predicted probability per class.
    p_y_given_x_2d = t.nnet.softmax(softmax2d_input / softmax_temperature)
    p_y_given_x_class_minor = p_y_given_x_2d.reshape((softmax_shape_input[0], softmax_shape_input[2],
                                                      softmax_shape_input[3], softmax_shape_input[4],
                                                      softmax_shape_input[1]))  # Result: batchSize, R,C,Z, Classes.
    p_y_given_x = p_y_given_x_class_minor.dimshuffle(0, 4, 1, 2, 3)  # Result: batchSize, Class, R, C, Z

    # Classification (EM) for each voxel
    y_pred = t.argmax(p_y_given_x, axis=1)  # Result: batchSize, R, C, Z

    return p_y_given_x, y_pred


# TODO: investigate this, maybe the part where it performs segmentation?
# Currently only used for pooling3d
def mirror_final_image_borders(image3d_bc012, final_borders_limit):
    bc012_mirror_pad = image3d_bc012
    for i in range(0, final_borders_limit[0]):
        bc012_mirror_pad = t.concatenate([bc012_mirror_pad, bc012_mirror_pad[:, :, -1:, :, :]], axis=2)
    # TODO: is this tensorflow tf.concat?
    for i in range(0, final_borders_limit[1]):
        bc012_mirror_pad = t.concatenate([bc012_mirror_pad, bc012_mirror_pad[:, :, :, -1:, :]], axis=3)

    for i in range(0, final_borders_limit[2]):
        bc012_mirror_pad = t.concatenate([bc012_mirror_pad, bc012_mirror_pad[:, :, :, :, -1:]], axis=4)

    return bc012_mirror_pad


# TODO: fix this stupid function with missing variables, ds and pooled_out
def pool_3d_mirror_pad(image3d_bc012, image3d_bc012_shape, pool_params):
    # image3d_bc012 dimensions: (batch, fms, r, c, z)
    # pool_params: [[dsr,dsc,dsz], [str_r,str_c,str_z], [mirrorPad-r,-c,-z], mode]
    ws = pool_params[0]  # window size
    stride = pool_params[1]  # stride
    mode1 = pool_params[3]  # max, sum, average_inc_pad, average_exc_pad
    bc012_mirror_pad = mirror_final_image_borders(image3d_bc012, pool_params[2])
    t.signal.pool.pool_3d(input=bc012_mirror_pad, ws=ws, ignore_border=True, st=stride, pad=(0, 0, 0), mode=mode1)

    # calculate the shape of the image after the max pooling.
    # This calculation is for ignore_border=True! Pooling should only be done in full areas in the mirror-padded image.
    new_image_shape = calc_shape_after_pooling(image3d_bc012_shape, pool_params, stride)
    return pooled_out, new_image_shape


def calc_shape_after_pooling(image3d_bc012_shape, pool_params, stride):
    return [image3d_bc012_shape[0], image3d_bc012_shape[1],
            int(ceil((image3d_bc012_shape[2] + pool_params[2][0] - ds[0] + 1) / (1.0 * stride[0]))),
            int(ceil((image3d_bc012_shape[3] + pool_params[2][1] - ds[1] + 1) / (1.0 * stride[1]))),
            int(ceil((image3d_bc012_shape[4] + pool_params[2][2] - ds[2] + 1) / (1.0 * stride[2])))]
