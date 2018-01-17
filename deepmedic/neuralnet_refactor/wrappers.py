class PathwaySamplingWrapper(object):
    """
    The API for these classes should resemble the API of Pathway and Cnn3d classes. But only what is needed by the
    sampling process of the training procedure.
    """

    # For CnnSamplingWrapper class.
    def __init__(self, pathway_instance):
        self._p_type = pathway_instance.p_type()
        self._sub_sampling_factor = pathway_instance.sub_sampling_factor()
        self._input_train_val_test_shape = pathway_instance.get_input_shape()
        self._output_train_val_test_shape = pathway_instance.get_output_shape()

    def p_type(self):
        return self._p_type

    def sub_sampling_factor(self):
        return self._sub_sampling_factor

    def get_input_shape(self):
        return self._input_train_val_test_shape

    def get_output_shape(self):
        return self._output_train_val_test_shape


# TODO: is this still needed with TensorFlow?
class CnnSamplingWrapper(object):
    """
    Only for the parallel process used during training. So that it won't re-load theano etc. There was a problem with
    cnmem when reloading theano.
    """

    def __init__(self, cnn3d_instance):
        batch_size = cnn3d_instance.batchSize
        batch_size_validation = cnn3d_instance.batchSizeValidation
        batch_size_testing = cnn3d_instance.batchSizeTesting

        output_shape_train = cnn3d_instance.finalTargetLayer.outputShapeTrain
        output_shape_val = cnn3d_instance.finalTargetLayer.outputShapeVal
        output_shape_test = cnn3d_instance.finalTargetLayer.outputShapeTest

        self.receptive_field_cnn = cnn3d_instance.recFieldCnn
        self.batch_size_train_val_test = [batch_size, batch_size_validation, batch_size_testing]
        self.final_target_layer_output_shape = [output_shape_train, output_shape_val, output_shape_test]
        self._num_pathways_with_input = cnn3d_instance.get_num_pathways_with_input()  # related pathways
        self.num_sub_sample_paths = cnn3d_instance.numSubsPaths
        self.pathways = []

        for pathway in cnn3d_instance.pathways:
            self.pathways.append(PathwaySamplingWrapper(pathway))

    def get_num_pathways_with_input(self):
        return self._num_pathways_with_input
