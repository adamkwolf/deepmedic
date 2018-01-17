import os
import gzip
import _pickle as pickle
import math


def load_from_file(filename_with_path):
    f = open(filename_with_path, 'rb')
    loaded_obj = pickle.load(f)
    f.close()

    return loaded_obj


def dump_to_file(my_obj, fn_with_path):
    """
    my_obj = object to pickle
    filename_with_path = a string with the full path+name

    The function uses the 'highest_protocol' which is supposed to be more storage efficient. It uses cPickle,
    which is coded in c and is supposed to be faster than pickle. Remember, this instance is safe to load only from a
    code which is fully-compatible (same version) ...with the code this was saved from, i.e. same classes define.
    """
    f = open(fn_with_path, 'wb')
    pickle.dump(my_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def load_from_gzip(fn_with_path):
    f = gzip.open(fn_with_path, 'rb')
    loaded_obj = pickle.load(f)
    f.close()

    return loaded_obj


def dump_to_gzip(my_obj, fn_with_path):
    f = gzip.open(fn_with_path, 'wb')
    pickle.dump(my_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def dump_cnn_to_gzip_save(cnn_instance, fn_with_path, logger=None):
    fn_with_path_save = os.path.abspath(fn_with_path + ".save")
    cnn_instance.freeGpuTrainingData()
    cnn_instance.freeGpuValidationData()
    cnn_instance.freeGpuTestingData()

    # Clear out the compiled functions, so that they are not saved with the instance:
    cnn_instance.cnnTrainModel = ""
    cnn_instance.cnnValidateModel = ""
    cnn_instance.cnnTestModel = ""
    cnn_instance.cnnVisualiseFmFunction = ""

    compiled_function_train = cnn_instance.cnnTrainModel
    compiled_function_test = cnn_instance.cnnTestModel
    compiled_function_val = cnn_instance.cnnValidateModel
    compiled_function_visualise = cnn_instance.cnnVisualiseFmFunction

    print_logger3("Saving network to: " + str(fn_with_path_save), logger)
    dump_to_gzip(cnn_instance, fn_with_path_save)
    print_logger3("Model saved.", logger)

    # Restore instance's values, which were cleared for the saving of the instance:
    cnn_instance.cnnTrainModel = compiled_function_train
    cnn_instance.cnnValidateModel = compiled_function_val
    cnn_instance.cnnTestModel = compiled_function_test
    cnn_instance.cnnVisualiseFmFunction = compiled_function_visualise

    return fn_with_path_save


def print_logger3(msg, logger=None):
    if logger is not None:
        logger.print3(msg)
    else:
        print(msg)


# Calculate the sub-sampled image part dimensions from image part size, patch size, and sub-sampled factor
def calculate_sub_sampled_image_part_dimensions(image_part_dimensions, patch_dimensions, sub_sample_factor):
    """
    This function gives you how big your sub_sampled-image-part should be, so that it corresponds to the correct
    number of central-voxels in the normal-part. Currently, it's coupled with the patch-size of the normal-scale.
    I.e. the sub_sampled-patch HAS TO BE THE SAME SIZE as the normal-scale, and corresponds to subFactor*patch_size
    in context. When the central voxels are not a multiple of the subFactor, you get ceil(), so +1 sub-patch. When
    the CNN repeats the pattern, it is giving dimension higher than the central-voxels of the normal-part,
    but then they are sliced-down to the correct number (in the cnn_make_model function, right after the repeat).
    This function works like this because of getImagePartFromSubsampledImageForTraining(), which gets a
    sub_sampled-image-part by going 1 normal-patch back from the top-left voxel of a normal-scale-part, and then 3
    ahead. If I change it to start from the top-left-CENTRAL-voxel back and front, I will be able to decouple the
    normal-patch size and the sub_sampled-patch-size.
    """

    # if patch is 17x17, a 17x17 sub_part is cool for 3 voxels with a sub_sample_factor.
    #  +2 to be ok for the 9x9 centrally classified voxels, so 19x19 sub-part.
    sub_sampled_image_part_dimensions = []
    for i in range(len(image_part_dimensions)):
        central_voxels = image_part_dimensions[i] - patch_dimensions[i] + 1
        central_voxels_sub_sampled_part = int(math.ceil(central_voxels * 1.0 / sub_sample_factor[i]))
        sub_sampled_image_size = patch_dimensions[i] + central_voxels_sub_sampled_part - 1
        sub_sampled_image_part_dimensions.append(sub_sampled_image_size)

    return sub_sampled_image_part_dimensions


def calculate_receptive_field(kernel_dimensions_per_layer_list):
    """
    Calculates the receptive field from kernel dimensions list per layer when strides is 1
    """
    if not kernel_dimensions_per_layer_list:
        return 0

    num_dimensions = len(kernel_dimensions_per_layer_list[0])
    receptive_field = [1] * num_dimensions

    for dimension in range(num_dimensions):
        for layer in range(len(kernel_dimensions_per_layer_list)):
            receptive_field[dimension] += kernel_dimensions_per_layer_list[layer][dimension] - 1

    return receptive_field


def check_receptive_field_vs_segment_size(receptive_field_dimension, segment_dimension):
    """
    Perform a check on the receptive field vs the segment size
    """
    num_receptive_field_dimensions = len(receptive_field_dimension)
    num_segment_dimensions = len(segment_dimension)

    if num_receptive_field_dimensions != num_segment_dimensions:
        print("ERROR: [in function check_receptive_field_vs_segment_size()] : Receptive field and image segment have "
              "different number of dimensions! (should be 3 for both! Exiting!)")
        exit(1)

    for dimension in range(num_receptive_field_dimensions):
        if receptive_field_dimension[dimension] > segment_dimension[dimension]:
            print("ERROR: [in function check_receptive_field_vs_segment_size()] : The segment-size (input) should be "
                  "at least as big as the receptive field of the model! This was not found to hold! Dimensions of "
                  "Receptive Field:", receptive_field_dimension, ". Dimensions of Segment: ", segment_dimension)
            return False

    return True


# check kernel dimension per layer correct 3d and number of layers
def check_kernel_dimensions_per_layer(kernel_dimensions_per_layer, num_layers):
    """
    :param kernel_dimensions_per_layer: a list with sub-lists. One sublist per layer. Each sublist should have 3
    integers, specifying the dimensions of the kernel at the corresponding layer of the pathway. eg:
    kernDimensionsPerLayer = [ [3,3,3], [3,3,3], [5,5,5] ]
    :param num_layers: the number of layers
    :return:
    """
    missing_kernel_dimensions = kernel_dimensions_per_layer is None
    wrong_num_layers = len(kernel_dimensions_per_layer) != num_layers

    if missing_kernel_dimensions or wrong_num_layers:
        return False

    return all(len(dimensions) != 3 for dimensions in kernel_dimensions_per_layer)


def is_sub_sample_factor_even(sub_sample_factor):
    return all(s % 2 != 1 for s in sub_sample_factor)
