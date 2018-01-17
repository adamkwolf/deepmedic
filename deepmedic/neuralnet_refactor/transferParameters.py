from deepmedic.neuralnet_refactor.pathwayTypes import PathwayTypes as Pt


def transfer_parameters_between_models(my_logger, cnn_target, cnn_source, layers_to_transfer):
    """
    :param my_logger: hello
    :param cnn_target: An instance of deepmedic.neural.net.cnn3d.Cnn3d. This is the model that
    will receive the parameters of the pre-trained model.
    :param cnn_source: Similar to the above. The parameters of
    this model will be transferred to the above.
    :param layers_to_transfer: A list of integers. The integers are
    the depth of the layers of cnnTarget that will be adopted from the pre-trained model. First layer is 1.
    Classification layer of the original 11-layers deep deepmedic is 11. The same layers from each parallel-pathway
    are transferred. If [] is given, no layer is transferred. If None is given, default functionality follows. Which
    transfers all layers except the classification layer. Example: In the original deepmedic, with 8 layers at each
    parallel path followed by 3 FC layers, [1,2,3,4,9,10] will transfer parameters of the 4 first layers of EACH
    parallel pathway, and 2 hidden FC layers (depth 9 and 10).
    """

    # Classification layer. NOT Softmax layer, which is not registered in the FC path. NOTE: Softmax layer HAS learnt
    # BIASES and I must transfer them separately if deepest pathway is asked to be transferred.
    deepest_layer_target_depth = len(cnn_target.pathways[0].getLayers()) + len(cnn_target.getFcPathway().getLayers())

    for i in range(len(cnn_target.pathways)):
        pathway = cnn_target.pathways[i]
        path_target_type = pathway.p_type()
        path_target_layers = pathway.getLayers()

        for layer_target in path_target_layers:
            layer_target_depth = layer_target + 1
            if not path_target_type:
                layer_target_depth += len(cnn_target.pathways[0].getLayers())

            # Check if this layer of Target should receive parameters from Source.
            # For list == None, we do the default transfer. Transfer all except the deepest classification Layer.
            has_transfer_layer = False

            if layers_to_transfer is None and layer_target_depth != deepest_layer_target_depth:
                has_transfer_layer = True

            if layers_to_transfer is not None and layer_target_depth in layers_to_transfer:
                has_transfer_layer = True

            if has_transfer_layer:
                my_logger.print3("[Pathway_" + str(pathway.getStringType()) + "][Conv.Layer_" + str(layer_target) +
                                 " (index)], depth [" + str(layer_target_depth) + "] (Target): Receiving parameters...")

                # Transfer stuff and get the correct Source path.
                if path_target_type != Pt.FC:
                    # if cnnSource has at least as many parallel pathways (-1 to exclude FC) as the number of the
                    # current pathwayTarget (+1 because it's index).
                    if (len(cnn_source.pathways) - 1) >= (i + 1):
                        path_source = cnn_source.pathways[i]
                    else:
                        path_source = cnn_source.pathways[-2]  # -1 is the FC pathway. -2 is the last parallel.
                        my_logger.print3("\t Source model has less parallel paths than Target. Parameters of Target "
                                         "are received from last parallel path of Source [Pathway_" +
                                         str(path_source.getStringType()) + "]")
                else:
                    path_source = cnn_source.getFcPathway()

                # Get the correct Source layer.
                if len(path_source.getLayers()) < layer_target + 1:
                    my_logger.print3("ERROR: This [Pathway_" + str(pathway.getStringType()) + "] of the [Source] model was found to have less layers than required!\n\t Number of layers in [Source] pathway: [" + str(len(path_source.getLayers())) + "].\n\t Number of layers in [Target] pathway: [" + str(len(pathway.getLayers())) + "].\n\t Tried to transfer parameters to [Target] layer with *index* in this pathway: [" + str(layer_target) + "]. (specified depth [" + str(layer_target_depth) + "]).\n\t Note: First layer of pathway has *index* [0].\n\t Note#2: To transfer parameters from a Source model with less layers than the Target, specify the depth of layers to transfer using the command line option [-layers].\n\t Try [-h] for help or see documentation.\nExiting!")
                    exit(1)

                layer_source = path_source.getLayers()[layer_target]
                my_logger.print3("\t ...receiving parameters from [Pathway_" + str(path_source.getStringType()) +
                                 "][Conv.Layer_" + str(layer_target) + " (index)] (Source).")

                # Found Source layer. Excellent. Now just transfer the parameters from Source to Target
                transfer_parameters_between_layers(my_logger=my_logger, layer_target=pathway, layer_source=layer_source)

                # It's the last Classification layer that was transferred. Also transfer the biases of the Softmax
                # layer, which is not in FC path.
                if layer_target_depth == deepest_layer_target_depth:
                    my_logger.print3("\t Last Classification layer was transfered. Thus for completeness, transfer "
                                     "the biases applied by the Softmax pseudo-layers. ")
                    soft_max_layer_target = cnn_target.finalTargetLayer
                    soft_max_layer_source = cnn_source.finalTargetLayer
                    # This should only transfer biases.
                    transfer_parameters_between_layers(my_logger, soft_max_layer_target, soft_max_layer_source)

    return cnn_target


# TODO: update these stupid names when refactoring cnn3d
def transfer_parameters_between_layers(my_logger, layer_target, layer_source):
    # VIOLATES _HIDDEN ENCAPSULATION! TEMPORARY TILL I FIX THE API (TILL AFTER DA).
    min_feature_maps = min(layer_target.getNumberOfFeatureMaps(), layer_source.getNumberOfFeatureMaps())
    min_input_chans = min(layer_target.inputShapeTrain[1], layer_source.inputShapeTrain[1])

    transfer_weights(layer_source, layer_target, min_feature_maps, min_input_chans, my_logger)
    transfer_biases(layer_source, layer_target, min_input_chans, my_logger)
    transfer_g_batch_norm(layer_source, layer_target, min_input_chans, my_logger)
    transfer_prelu(layer_source, layer_target, min_input_chans, my_logger)

    # For the rolling average used in inference by Batch-Norm.
    layer_target_rolling_average = layer_target._rollingAverageForBatchNormalizationOverThatManyBatches
    layer_source_rolling_average = layer_source._rollingAverageForBatchNormalizationOverThatManyBatches
    min_length_rolling_average = min(layer_target_rolling_average, layer_source_rolling_average)

    transfer_mu_batch_norm_averages(layer_source, layer_target, min_input_chans, min_length_rolling_average, my_logger)
    transfer_var_batch_norm_averages(layer_source, layer_target, min_input_chans, min_length_rolling_average, my_logger)


# TODO: update these stupid names when refactoring cnn3d
def transfer_var_batch_norm_averages(layer_source, layer_target, min_input_chans, min_length_rolling_average, my_logger):
    layer_target_variance_batch_norm = layer_target._varBnsArrayForRollingAverage is not None
    layer_source_variance_batch_norm = layer_source._varBnsArrayForRollingAverage is not None
    if layer_target_variance_batch_norm and layer_source_variance_batch_norm:
        my_logger.print3("\t Transferring rolling average of Variance of Batch Norm [varBnsArrayForRollingAverage].")
        target_value = layer_target._varBnsArrayForRollingAverage.get_value()
        source_value = layer_source._varBnsArrayForRollingAverage.get_value()
        target_value[:min_length_rolling_average, :min_input_chans] = \
            source_value[:min_length_rolling_average, :min_input_chans]
        layer_target._varBnsArrayForRollingAverage.set_value(target_value)


# TODO: update these stupid names when refactoring cnn3d
def transfer_mu_batch_norm_averages(layer_source, layer_target, min_input_chans, min_length_rolling_average, my_logger):
    has_layer_target_batch_norm_rolling_average = layer_target._muBnsArrayForRollingAverage is not None
    has_layer_source_batch_norm_rolling_average = layer_source._muBnsArrayForRollingAverage is not None
    if has_layer_target_batch_norm_rolling_average and has_layer_source_batch_norm_rolling_average:
        my_logger.print3("\t Transferring rolling average of MU of Batch Norm [muBnsArrayForRollingAverage].")
        target_value = layer_target._muBnsArrayForRollingAverage.get_value()
        source_value = layer_source._muBnsArrayForRollingAverage.get_value()
        target_value[:min_length_rolling_average, :min_input_chans] = \
            source_value[:min_length_rolling_average, :min_input_chans]
        layer_target._muBnsArrayForRollingAverage.set_value(target_value)


# TODO: update these stupid names when refactoring cnn3d
def transfer_weights(layer_source, layer_target, min_feature_maps, min_input_chans, my_logger):
    has_layer_target_w = layer_target._W is not None
    has_layer_source_w = layer_source._W is not None
    if has_layer_target_w and has_layer_source_w:
        my_logger.print3("\t Transferring weights [W].")
        target_value = layer_target._W.get_value()
        source_value = layer_source._W.get_value()
        target_value[:min_feature_maps, :min_input_chans, :, :, :] = \
            source_value[:min_feature_maps, :min_input_chans, :, :, :]
        layer_target._W.set_value(target_value)


# TODO: update these stupid names when refactoring cnn3d
def transfer_biases(layer_source, layer_target, min_input_chans, my_logger):
    has_layer_target_b = layer_target._b is not None
    has_layer_source_b = layer_source._b is not None
    if has_layer_target_b and has_layer_source_b:
        my_logger.print3("\t Transferring biases [b].")
        target_value = layer_target._b.get_value()
        source_value = layer_source._b.get_value()
        target_value[:min_input_chans] = source_value[:min_input_chans]
        layer_target._b.set_value(target_value)


# TODO: update these stupid names when refactoring cnn3d
def transfer_g_batch_norm(layer_source, layer_target, min_input_chans, my_logger):
    has_layer_target_gBn = layer_target._gBn is not None
    has_layer_source_gBn = layer_source._gBn is not None
    if has_layer_target_gBn and has_layer_source_gBn:
        my_logger.print3("\t Transferring g of Batch Norm [gBn].")
        target_value = layer_target._gBn.get_value()
        source_value = layer_source._gBn.get_value()
        target_value[:min_input_chans] = source_value[:min_input_chans]
        layer_target._gBn.set_value(target_value)


# TODO: update these stupid names when refactoring cnn3d
def transfer_prelu(layer_source, layer_target, min_input_chans, my_logger):
    has_layer_target_a_prelu = layer_target._aPrelu is not None
    has_layer_source_a_prelu = layer_source._aPrelu is not None
    if has_layer_target_a_prelu and has_layer_source_a_prelu:
        my_logger.print3("\t Transferring a of PReLu [aPrelu].")
        target_value = layer_target._aPrelu.get_value()
        source_value = layer_source._aPrelu.get_value()
        target_value[:min_input_chans] = source_value[:min_input_chans]
        layer_target._aPrelu.set_value(target_value)
