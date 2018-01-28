from six.moves import xrange
import numpy as np
from math import ceil

import theano.tensor as t

from deepmedic.neuralnet.pathwayTypes import PathwayTypes
from deepmedic.neuralnet.layers import ConvLayer, LowRankConvLayer
from deepmedic.neuralnet.utils import calcRecFieldFromKernDimListPerLayerWhenStrides1


#################################################################
#                         Pathway Types                         #
#################################################################

def crop_dim_array(to_crop, to_match):
    # to_match : [ batch size, num of fms, r, c, z]
    return to_crop[:, :, :to_match[2], :to_match[3], :to_match[4]]


def repeat_by_factor(array_5d, factor_3d):
    # array_5d: [batch size, num of FMs, r, c, z]. Ala input/output of conv layers.
    # Repeat FM in the three last dimensions, to upsample back to the normal resolution space.
    expanded_r = array_5d.repeat(factor_3d[0], axis=2)
    expanded_rc = expanded_r.repeat(factor_3d[1], axis=3)
    expanded_rcz = expanded_rc.repeat(factor_3d[2], axis=4)
    return expanded_rcz


def upsample_array(to_upsample, upsample_factor, scheme="repeat", dimensions=None):
    # to_upsample : [batch_size, numberOfFms, r, c, z].
    output = None
    if scheme == "repeat":
        output = repeat_by_factor(to_upsample, upsample_factor)
    else:
        print("ERROR: in upsample_array(...). Not implemented type of upsampling! Exiting!")
        exit(1)

    if dimensions is not None:
        # If the central-voxels are eg 10, the upsampled-part will have 4 central voxels. Which above will be
        # repeated to 3*4 = 12. I need to clip the last ones, to have the same dimension as the input from 1st
        # pathway, which will have dimensions equal to the centrally predicted voxels (10)
        output = crop_dim_array(output, dimensions)

    return output


def get_middle_feature_map(fms, num_central_voxels):
    # fms: a 5D tensor, [batch, fms, r, c, z]
    feature_map_shape = t.shape(fms)  # fms.shape works too, but this is clearer theano grammar.

    # if part is of even width, one voxel to the left is the centre.
    r_center_index = (feature_map_shape[2] - 1) // 2
    r_start_index = r_center_index - (num_central_voxels[0] - 1) // 2
    r_end_index = r_start_index + num_central_voxels[0]  # Excluding
    c_center_index = (feature_map_shape[3] - 1) // 2
    c_start_index = c_center_index - (num_central_voxels[1] - 1) // 2
    c_end_index = c_start_index + num_central_voxels[1]  # Excluding

    if len(num_central_voxels) == 2:  # the input FMs are of 2 dimensions (for future use)
        return fms[:, :, r_start_index: r_end_index, c_start_index: c_end_index]
    elif len(num_central_voxels) == 3:  # the input FMs are of 3 dimensions
        z_center_index = (feature_map_shape[4] - 1) // 2
        z_start_index = z_center_index - (num_central_voxels[2] - 1) // 2
        z_end_index = z_start_index + num_central_voxels[2]  # Excluding
        return fms[:, :, r_start_index: r_end_index, c_start_index: c_end_index, z_start_index: z_end_index]
    else:  # wrong number of dimensions!
        return -1


def make_connection_between_layers(logger, deep_output_tr_val_test, deep_output_tr_val_test_shape,
                                   earlier_output_tr_val_test, earlier_output_tr_val_test_shape):
    # Add the outputs of the two layers and return the output, as well as its dimensions. Result: The result should
    # have exactly the same shape as the output of the Deeper layer. Both #FMs and Dimensions of FMs.

    (deep_train, deep_val, deep_test) = deep_output_tr_val_test
    (deep_train_shape, deep_val_shape, deep_test_shape) = deep_output_tr_val_test_shape
    (earlier_train, earlier_val, earlier_test) = earlier_output_tr_val_test
    (earlier_train_shape, earlier_val_shape, earlier_test_shape) = earlier_output_tr_val_test_shape
    # Note: deep_train_shape has dimensions: [batchSize, FMs, r, c, z] The deeper FMs can be greater
    # only when there is upsampling. But then, to do residuals, I would need to upsample the earlier FMs. Not
    # implemented.

    deeper_train_than_earlier = np.any(np.asarray(deep_train_shape[2:]) > np.asarray(earlier_train_shape[2:]))
    deeper_val_than_earlier = np.any(np.asarray(deep_val_shape[2:]) > np.asarray(earlier_val_shape[2:]))
    deeper_test_than_earlier = np.any(np.asarray(deep_test_shape[2:]) > np.asarray(earlier_test_shape[2:]))

    if deeper_train_than_earlier or deeper_val_than_earlier or deeper_test_than_earlier:
        logger.print3("ERROR: In function [make_connection_between_layers] the RCZ-dimensions of a deeper layer FMs "
                      "were found greater than the earlier layers. Not implemented functionality. Exiting!")

        logger.print3("\t (train) Dimensions of Deeper Layer=" + str(deep_train_shape) +
                      ". Dimensions of Earlier Layer=" + str(earlier_train_shape))

        logger.print3("\t (val) Dimensions of Deeper Layer=" + str(deep_val_shape) + ". Dimensions of Earlier Layer="
                      + str(earlier_val_shape))

        logger.print3("\t (test) Dimensions of Deeper Layer=" + str(deep_test_shape) + ". Dimensions of Earlier Layer="
                      + str(earlier_test_shape))
        exit(1)

    # get the part of the earlier layer that is of the same dimensions as the FMs of the deeper:
    earlier_train_part = get_middle_feature_map(earlier_train, deep_train_shape[2:])
    earlier_val_part = get_middle_feature_map(earlier_val, deep_val_shape[2:])
    earlier_test_part = get_middle_feature_map(earlier_test, deep_test_shape[2:])

    # Add the FMs, after taking care of zero padding if the deeper layer has more FMs.
    num_deeper_fms = deep_train_shape[1]
    num_earlier_fms = earlier_train_shape[1]
    if num_deeper_fms >= num_earlier_fms:
        res_conn_train = t.inc_subtensor(deep_train[:, :num_earlier_fms, :, :, :], earlier_train_part, inplace=False)
        res_conn_val = t.inc_subtensor(deep_val[:, :num_earlier_fms, :, :, :], earlier_val_part, inplace=False)
        res_conn_test = t.inc_subtensor(deep_test[:, :num_earlier_fms, :, :, :], earlier_test_part, inplace=False)
    else:  # Deeper FMs are fewer than earlier. This should not happen in most architectures. But oh well...
        res_conn_train = deep_train + earlier_train_part[:, :num_deeper_fms, :, :, :]
        res_conn_val = deep_val + earlier_val_part[:, :num_deeper_fms, :, :, :]
        res_conn_test = deep_test + earlier_test_part[:, :num_deeper_fms, :, :, :]

    # Dimensions of output are the same as those of the deeperLayer
    return res_conn_train, res_conn_val, res_conn_test


#################################################################
#                        Classes of Pathways                    #
#################################################################

class Pathway(object):
    # This is a virtual class.

    def __init__(self, p_name=None):
        self._pName = p_name
        self._pType = None  # Pathway Type.

        # === Input to the pathway ===
        self._inputTrain = None
        self._inputVal = None
        self._inputTest = None
        self._inputShapeTrain = None
        self._inputShapeVal = None
        self._inputShapeTest = None

        # === Basic architecture parameters ===
        self._layersInPathway = []
        self._subsFactor = [1, 1, 1]
        self._recField = None  # At the end of pathway

        # === Output of the block ===
        self._outputTrain = None
        self._outputVal = None
        self._outputTest = None
        self._outputShapeTrain = None
        self._outputShapeVal = None
        self._outputShapeTest = None

    def makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(self,
                                                             myLogger,

                                                             inputTrain,
                                                             inputVal,
                                                             inputTest,
                                                             inputDimsTrain,
                                                             inputDimsVal,
                                                             inputDimsTest,

                                                             numKernsPerLayer,
                                                             kernelDimsPerLayer,

                                                             convWInitMethod,
                                                             useBnPerLayer,
                                                             # As a flag for case that I want to apply BN on input image. I want to apply to input of FC.
                                                             rollingAverageForBatchNormalizationOverThatManyBatches,
                                                             activFuncPerLayer,
                                                             dropoutRatesPerLayer=[],

                                                             poolingParamsStructureForThisPathwayType=[],

                                                             indicesOfLowerRankLayersForPathway=[],
                                                             ranksOfLowerRankLayersForPathway=[],

                                                             indicesOfLayersToConnectResidualsInOutputForPathway=[]
                                                             ):
        rng = np.random.RandomState(55789)
        myLogger.print3("[Pathway_" + str(self.getStringType()) + "] is being built...")

        self._recField = self.calcRecFieldOfPathway(kernelDimsPerLayer)

        self._setInputAttributes(inputTrain, inputVal, inputTest, inputDimsTrain, inputDimsVal, inputDimsTest)
        myLogger.print3(
            "\t[Pathway_" + str(self.getStringType()) + "]: Input's Shape: (Train) " + str(self._inputShapeTrain) + \
            ", (Val) " + str(self._inputShapeVal) + ", (Test) " + str(self._inputShapeTest))

        inputToNextLayerTrain = self._inputTrain;
        inputToNextLayerVal = self._inputVal;
        inputToNextLayerTest = self._inputTest
        inputToNextLayerShapeTrain = self._inputShapeTrain;
        inputToNextLayerShapeVal = self._inputShapeVal;
        inputToNextLayerShapeTest = self._inputShapeTest
        numOfLayers = len(numKernsPerLayer)
        for layer_i in xrange(0, numOfLayers):
            thisLayerFilterShape = [numKernsPerLayer[layer_i], inputToNextLayerShapeTrain[1]] + kernelDimsPerLayer[
                layer_i]

            thisLayerUseBn = useBnPerLayer[layer_i]
            thisLayerActivFunc = activFuncPerLayer[layer_i]
            thisLayerDropoutRate = dropoutRatesPerLayer[layer_i] if dropoutRatesPerLayer else 0

            thisLayerPoolingParameters = poolingParamsStructureForThisPathwayType[layer_i]

            myLogger.print3("\t[Conv.Layer_" + str(layer_i) + "], Filter Shape: " + str(thisLayerFilterShape))
            myLogger.print3(
                "\t[Conv.Layer_" + str(layer_i) + "], Input's Shape: (Train) " + str(inputToNextLayerShapeTrain) + \
                ", (Val) " + str(inputToNextLayerShapeVal) + ", (Test) " + str(inputToNextLayerShapeTest))

            if layer_i in indicesOfLowerRankLayersForPathway:
                layer = LowRankConvLayer(
                    ranksOfLowerRankLayersForPathway[indicesOfLowerRankLayersForPathway.index(layer_i)])
            else:  # normal conv layer
                layer = ConvLayer()
            layer.makeLayer(rng,
                            inputToLayerTrain=inputToNextLayerTrain,
                            inputToLayerVal=inputToNextLayerVal,
                            inputToLayerTest=inputToNextLayerTest,
                            inputToLayerShapeTrain=inputToNextLayerShapeTrain,
                            inputToLayerShapeVal=inputToNextLayerShapeVal,
                            inputToLayerShapeTest=inputToNextLayerShapeTest,

                            filterShape=thisLayerFilterShape,
                            poolingParameters=thisLayerPoolingParameters,
                            convWInitMethod=convWInitMethod,
                            useBnFlag=thisLayerUseBn,
                            rollingAverageForBatchNormalizationOverThatManyBatches=rollingAverageForBatchNormalizationOverThatManyBatches,
                            activationFunc=thisLayerActivFunc,
                            dropoutRate=thisLayerDropoutRate
                            )
            self._layersInPathway.append(layer)

            if layer_i not in indicesOfLayersToConnectResidualsInOutputForPathway:  # not a residual connecting here
                inputToNextLayerTrain = layer.outputTrain
                inputToNextLayerVal = layer.outputVal
                inputToNextLayerTest = layer.outputTest
            else:  # make residual connection
                myLogger.print3("\t[Pathway_" + str(
                    self.getStringType()) + "]: making Residual Connection between output of [Layer_" + str(
                    layer_i) + "] to input of previous layer.")
                deeperLayerOutputImagesTrValTest = (layer.outputTrain, layer.outputVal, layer.outputTest)
                deeperLayerOutputImageShapesTrValTest = (
                    layer.outputShapeTrain, layer.outputShapeVal, layer.outputShapeTest)
                assert layer_i > 0  # The very first layer (index 0), should never be provided for now. Cause I am connecting 2 layers back.
                earlierLayer = self._layersInPathway[layer_i - 1]
                earlierLayerOutputImagesTrValTest = (
                    earlierLayer.inputTrain, earlierLayer.inputVal, earlierLayer.inputTest)
                earlierLayerOutputImageShapesTrValTest = (
                    earlierLayer.inputShapeTrain, earlierLayer.inputShapeVal, earlierLayer.inputShapeTest)

                (inputToNextLayerTrain,
                 inputToNextLayerVal,
                 inputToNextLayerTest) = make_connection_between_layers(myLogger, deeperLayerOutputImagesTrValTest,
                                                                        deeperLayerOutputImageShapesTrValTest,
                                                                        earlierLayerOutputImagesTrValTest,
                                                                        earlierLayerOutputImageShapesTrValTest)
                layer.outputAfterResidualConnIfAnyAtOutpTrain = inputToNextLayerTrain
                layer.outputAfterResidualConnIfAnyAtOutpVal = inputToNextLayerVal
                layer.outputAfterResidualConnIfAnyAtOutpTest = inputToNextLayerTest
            # Residual connections preserve the both the number of FMs and the dimensions of the FMs, the same as in the later, deeper layer.
            inputToNextLayerShapeTrain = layer.outputShapeTrain
            inputToNextLayerShapeVal = layer.outputShapeVal
            inputToNextLayerShapeTest = layer.outputShapeTest

        self._setOutputAttributes(inputToNextLayerTrain, inputToNextLayerVal, inputToNextLayerTest,
                                  inputToNextLayerShapeTrain, inputToNextLayerShapeVal, inputToNextLayerShapeTest)

        myLogger.print3(
            "\t[Pathway_" + str(self.getStringType()) + "]: Output's Shape: (Train) " + str(self._outputShapeTrain) + \
            ", (Val) " + str(self._outputShapeVal) + ", (Test) " + str(self._outputShapeTest))

        myLogger.print3("[Pathway_" + str(self.getStringType()) + "] done.")

    # Skip connections to end of pathway.
    def makeMultiscaleConnectionsForLayerType(self, convLayersToConnectToFirstFcForMultiscaleFromThisLayerType):

        layersInThisPathway = self.getLayers()

        [outputOfPathwayTrain, outputOfPathwayVal, outputOfPathwayTest] = self.getOutput()
        [outputShapeTrain, outputShapeVal, outputShapeTest] = self.getShapeOfOutput()
        numOfCentralVoxelsToGetTrain = outputShapeTrain[2:];
        numOfCentralVoxelsToGetVal = outputShapeVal[2:];
        numOfCentralVoxelsToGetTest = outputShapeTest[2:]

        for convLayer_i in convLayersToConnectToFirstFcForMultiscaleFromThisLayerType:
            thisLayer = layersInThisPathway[convLayer_i]

            middlePartOfFmsTrain = get_middle_feature_map(thisLayer.outputTrain, numOfCentralVoxelsToGetTrain)
            middlePartOfFmsVal = get_middle_feature_map(thisLayer.outputVal, numOfCentralVoxelsToGetVal)
            middlePartOfFmsTest = get_middle_feature_map(thisLayer.outputTest, numOfCentralVoxelsToGetTest)

            outputOfPathwayTrain = t.concatenate([outputOfPathwayTrain, middlePartOfFmsTrain], axis=1)
            outputOfPathwayVal = t.concatenate([outputOfPathwayVal, middlePartOfFmsVal], axis=1)
            outputOfPathwayTest = t.concatenate([outputOfPathwayTest, middlePartOfFmsTest], axis=1)
            outputShapeTrain[1] += thisLayer.getNumberOfFeatureMaps();
            outputShapeVal[1] += thisLayer.getNumberOfFeatureMaps();
            outputShapeTest[1] += thisLayer.getNumberOfFeatureMaps();

        self._setOutputAttributes(outputOfPathwayTrain, outputOfPathwayVal, outputOfPathwayTest,
                                  outputShapeTrain, outputShapeVal, outputShapeTest)

    # The below should be updated, and calculated in here properly with private function and per layer.
    def calcRecFieldOfPathway(self, kernelDimsPerLayer):
        return calcRecFieldFromKernDimListPerLayerWhenStrides1(kernelDimsPerLayer)

    def calcInputRczDimsToProduceOutputFmsOfCompatibleDims(self, thisPathWayKernelDims, dimsOfOutputFromPrimaryPathway):
        recFieldAtEndOfPathway = self.calcRecFieldOfPathway(thisPathWayKernelDims)
        rczDimsOfInputToPathwayShouldBe = [-1, -1, -1]
        rczDimsOfOutputOfPathwayShouldBe = [-1, -1, -1]

        rczDimsOfOutputFromPrimaryPathway = dimsOfOutputFromPrimaryPathway[2:]
        for rcz_i in xrange(3):
            rczDimsOfOutputOfPathwayShouldBe[rcz_i] = int(
                ceil(rczDimsOfOutputFromPrimaryPathway[rcz_i] / (1.0 * self.subsFactor()[rcz_i])))
            rczDimsOfInputToPathwayShouldBe[rcz_i] = recFieldAtEndOfPathway[rcz_i] + rczDimsOfOutputOfPathwayShouldBe[
                rcz_i] - 1
        return rczDimsOfInputToPathwayShouldBe

    # Setters
    def _setInputAttributes(self, inputToLayerTrain, inputToLayerVal, inputToLayerTest, inputToLayerShapeTrain,
                            inputToLayerShapeVal, inputToLayerShapeTest):
        self._inputTrain = inputToLayerTrain;
        self._inputVal = inputToLayerVal;
        self._inputTest = inputToLayerTest
        self._inputShapeTrain = inputToLayerShapeTrain;
        self._inputShapeVal = inputToLayerShapeVal;
        self._inputShapeTest = inputToLayerShapeTest

    def _setOutputAttributes(self, outputTrain, outputVal, outputTest, outputShapeTrain, outputShapeVal,
                             outputShapeTest):
        self._outputTrain = outputTrain;
        self._outputVal = outputVal;
        self._outputTest = outputTest
        self._outputShapeTrain = outputShapeTrain;
        self._outputShapeVal = outputShapeVal;
        self._outputShapeTest = outputShapeTest

    # Getters
    def pName(self):
        return self._pName

    def pType(self):
        return self._pType

    def getLayers(self):
        return self._layersInPathway

    def getLayer(self, index):
        return self._layersInPathway[index]

    def subsFactor(self):
        return self._subsFactor

    def getOutput(self):
        return [self._outputTrain, self._outputVal, self._outputTest]

    def getShapeOfOutput(self):
        return [self._outputShapeTrain, self._outputShapeVal, self._outputShapeTest]

    def getShapeOfInput(self):
        return [self._inputShapeTrain, self._inputShapeVal, self._inputShapeTest]

    # Other API :
    def getStringType(self):
        raise NotImplementedMethod()  # Abstract implementation. Children classes should implement this.

    # Will be overriden for lower-resolution pathways.
    def getOutputAtNormalRes(self):
        return self.getOutput()

    def getShapeOfOutputAtNormalRes(self):
        return self.getShapeOfOutput()


class NormalPathway(Pathway):
    def __init__(self, p_name=None):
        Pathway.__init__(self, p_name)
        self._pType = PathwayTypes.NORM

    # Override parent's abstract classes.
    def getStringType(self):
        return "NORMAL"


class SubsampledPathway(Pathway):
    def __init__(self, subsamplingFactor, p_name=None):
        Pathway.__init__(self, p_name)
        self._pType = PathwayTypes.SUBS
        self._subsFactor = subsamplingFactor

        self._outputNormResTrain = None
        self._outputNormResVal = None
        self._outputNormResTest = None
        self._outputNormResShapeTrain = None
        self._outputNormResShapeVal = None
        self._outputNormResShapeTest = None

    def upsampleOutputToNormalRes(self, upsamplingScheme="repeat",
                                  shapeToMatchInRczTrain=None, shapeToMatchInRczVal=None, shapeToMatchInRczTest=None):
        # should be called only once to build. Then just call getters if needed to get upsampled layer again.
        [outputTrain, outputVal, outputTest] = self.getOutput()
        [outputShapeTrain, outputShapeVal, outputShapeTest] = self.getShapeOfOutput()

        outputNormResTrain = upsample_array(outputTrain,
                                            self.subsFactor(),
                                            upsamplingScheme,
                                            shapeToMatchInRczTrain)
        outputNormResVal = upsample_array(outputVal,
                                          self.subsFactor(),
                                          upsamplingScheme,
                                          shapeToMatchInRczVal)
        outputNormResTest = upsample_array(outputTest,
                                           self.subsFactor(),
                                           upsamplingScheme,
                                           shapeToMatchInRczTest)

        outputNormResShapeTrain = outputShapeTrain[:2] + shapeToMatchInRczTrain[2:]
        outputNormResShapeVal = outputShapeVal[:2] + shapeToMatchInRczVal[2:]
        outputNormResShapeTest = outputShapeTest[:2] + shapeToMatchInRczTest[2:]

        self._setOutputAttributesNormRes(outputNormResTrain, outputNormResVal, outputNormResTest,
                                         outputNormResShapeTrain, outputNormResShapeVal, outputNormResShapeTest)

    def _setOutputAttributesNormRes(self, outputNormResTrain, outputNormResVal, outputNormResTest,
                                    outputNormResShapeTrain, outputNormResShapeVal, outputNormResShapeTest):
        # Essentially this is after the upsampling "layer"
        self._outputNormResTrain = outputNormResTrain;
        self._outputNormResVal = outputNormResVal;
        self._outputNormResTest = outputNormResTest
        self._outputNormResShapeTrain = outputNormResShapeTrain;
        self._outputNormResShapeVal = outputNormResShapeVal;
        self._outputNormResShapeTest = outputNormResShapeTest

    # OVERRIDING parent's classes.
    def getStringType(self):
        return "SUBSAMPLED" + str(self.subsFactor())

    def getOutputAtNormalRes(self):
        # upsampleOutputToNormalRes() must be called first once.
        return [self._outputNormResTrain, self._outputNormResVal, self._outputNormResTest]

    def getShapeOfOutputAtNormalRes(self):
        # upsampleOutputToNormalRes() must be called first once.
        return [self._outputNormResShapeTrain, self._outputNormResShapeVal, self._outputNormResShapeTest]


class FcPathway(Pathway):
    def __init__(self, p_name=None):
        Pathway.__init__(self, p_name)
        self._pType = PathwayTypes.FC

    # Override parent's abstract classes.
    def getStringType(self):
        return "FC"
