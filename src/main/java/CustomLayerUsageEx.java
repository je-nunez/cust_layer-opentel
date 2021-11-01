/*******************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// package org.deeplearning4j.examples.advanced.features.customizingdl4j.layers;

// import org.deeplearning4j.examples.advanced.features.customizingdl4j.layers.layer.CustomLayer;
import layer.CustomLayer;

import org.apache.commons.io.FilenameUtils;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.gradientcheck.GradientCheckUtil;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;


// import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.lossfunctions.LossFunctions;
// import org.nd4j.systeminfo.SystemInfo;
// import org.nd4j.versioncheck.VersionCheck;

import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Scope;
// import io.opentelemetry.extension.annotations.WithSpan;
import io.opentelemetry.sdk.autoconfigure.OpenTelemetrySdkAutoConfiguration;

import java.io.File;
import java.io.IOException;
import java.time.Instant;
import java.util.List;
import java.util.Random;

/**
 * Custom layer example, instrumented with OpenTelemetry.
 *
 * @author Jose E. Nunez (based on original code of DL4J from Alex Black)
 */
public class CustomLayerUsageEx {

    static {
        //Double precision for the gradient checks. See comments in the doGradientCheck() method
        // See also http://nd4j.org/userguide.html#miscdatatype
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    // for MNIST:
    private static final int outputNum = 10;  // Number of possible outcomes
    private static final int batchSize = 64;  // Test batch size
    private static DataSetIterator mnistTrain;
    private static DataSetIterator mnistTest;
    private static boolean setListenersDuringTraining = false;

    // for OpenTelemetry:
    private static Tracer tracer;

    public static void main(String[] args) throws IOException {

        setupOpenTelemetry();

        Span span =
             tracer.spanBuilder("mainApp")
                 .setStartTimestamp(Instant.now())
                 .setAttribute("thread.id", Thread.currentThread().getId())
                 .setAttribute("telemetry.sdk.language", "java")
                 .setAttribute("os.type",
                               System.getProperty("os.name").toLowerCase())
                 .setAttribute("os.version",
                               System.getProperty("os.version"))
                 // The above are semantic attributes
                 .startSpan();

        try (Scope scope = span.makeCurrent()) {
            runInitialTests();
            doGradientCheck();

            // prepare for MNIST:
            double bestScore = Double.MIN_VALUE;
            Activation bestFirstScoreActiv = Activation.RELU;
            Activation bestSecondaryScoreActiv = Activation.RELU;

            mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
            mnistTest = new MnistDataSetIterator(batchSize, false, 12345);

            // This hyper-parameter optimization is better done with
            // DL4J Arbiter
            for (Activation firstActiv: Activation.values()) {

                for (Activation secondActiv: Activation.values()) {

                    double currScore = runMnistTest(firstActiv, secondActiv);
                    if (bestScore < currScore) {
                        bestScore = currScore;
                        bestFirstScoreActiv = firstActiv;
                        bestSecondaryScoreActiv = secondActiv;
                    }
                }
            }
            System.out.printf("---- Best activation: %s %s: accuracy: %.9f\n",
                              bestFirstScoreActiv, bestSecondaryScoreActiv,
                              bestScore);
        } finally {
            span.end();
        }
    }


    private static void setupOpenTelemetry() {
        OpenTelemetry openTelemetry = OpenTelemetrySdkAutoConfiguration.initialize();

        tracer = openTelemetry.getTracer("CustomLayer");
    }


    private static void runInitialTests() throws IOException {
        /*
        This method shows the configuration and use of the custom layer.
        It also shows some basic sanity checks and tests for the layer.
        In practice, these tests should be implemented as unit tests; for simplicity, we are just printing the results
         */

        Span span =
             tracer.spanBuilder("runInitialTests")
                 .setStartTimestamp(Instant.now())
                 .setAttribute("thread.id", Thread.currentThread().getId())
                 // The above is a semantic attribute
                 .startSpan();

        try (Scope scope = span.makeCurrent()) {
            System.out.println("---- Starting Initial Tests -----");

            int nIn = 5;
            int nOut = 8;

            // Let's create a network with our custom layer

            MultiLayerConfiguration config =
                       new NeuralNetConfiguration.Builder()
                           .updater(new RmsProp(0.95))
                           .weightInit(WeightInit.XAVIER)
                           .l2(0.03)
                           .list()
                           .layer(0, new DenseLayer.Builder()
                               .activation(Activation.TANH)
                               .nIn(nIn)
                               .nOut(6)
                               .build())
                           .layer(1, new CustomLayer.Builder() // Custom layer
                               .activation(Activation.TANH)
                               .secondActivationFunction(Activation.SIGMOID)
                               .nIn(6)
                               .nOut(7)
                               .build())
                           .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                               .activation(Activation.SOFTMAX)
                               .nIn(7)
                               .nOut(nOut)
                               .build())
                           .build();

            // First:  run some basic sanity checks on the configuration:
            double customLayerL2 = ((L2Regularization)
                                       ((BaseLayer)config.getConf(1).getLayer())
                                           .getRegularization()
                                           .get(0)
                                   )
                                      .getL2()
                                      .valueAt(0, 0);
            System.out.printf("l2 coefficient for custom layer: %f\n",
                              customLayerL2);
            IUpdater customLayerUpdater =
                             ((BaseLayer)config
                                 .getConf(1).getLayer()
                             ).getIUpdater();
            System.out.printf("Updater for custom layer: %s\n",
                              customLayerUpdater);

            // Second: We need to ensure that that the JSON and YAML
            // configuration works, with the custom layer
            // If there were problems with serialization, you'd get an
            // exception during deserialization
            // ("No suitable constructor found..." for example)
            String configAsJson = config.toJson();
            String configAsYaml = config.toYaml();
            MultiLayerConfiguration fromJson = MultiLayerConfiguration.fromJson(configAsJson);
            MultiLayerConfiguration fromYaml = MultiLayerConfiguration.fromYaml(configAsYaml);

            System.out.println("JSON configuration works: " + config.equals(fromJson));
            System.out.println("YAML configuration works: " + config.equals(fromYaml));

            MultiLayerNetwork net = new MultiLayerNetwork(config);
            net.init();

            // Third: Let's run some more basic tests. First, check that the
            // forward and backward pass methods don't throw any exceptions
            // To do this: we'll create some simple test data
            int minibatchSize = 5;
            INDArray testFeatures = Nd4j.rand(minibatchSize, nIn);
            INDArray testLabels = Nd4j.zeros(minibatchSize, nOut);
            Random r = new Random(12345);
            for (int i = 0; i < minibatchSize; i++) {
                // Random one-hot labels data
                testLabels.putScalar(i, r.nextInt(nOut), 1);
            }

            List<INDArray> activations = net.feedForward(testFeatures);
            INDArray activationsCustomLayer = activations.get(2);   //Activations of layer 2 (0 is input layer)
            System.out.println("\nActivations from custom layer:");
            System.out.println(activationsCustomLayer);
            net.fit(new DataSet(testFeatures, testLabels));

            // Finally, let's check the model serialization process, using
            // ModelSerializer:
            net.save(new File("CustomLayerModel.zip"), true);
            MultiLayerNetwork restored = MultiLayerNetwork.load(new File("CustomLayerModel.zip"), true);

            System.out.println();
            System.out.println("Original and restored networks: configs are equal: "
                               + net.getLayerWiseConfigurations().equals(
                                          restored.getLayerWiseConfigurations()
                                         )
                              );
            System.out.println("Original and restored networks: parameters are equal: "
                               + net.params().equals(restored.params())
                              );
        } finally {
            span.end();
        }
    }


    private static void doGradientCheck() {
        /*
        Gradient checks are one of the most important components of implementing a layer
        They are necessary to ensure that your implementation is correct: without them, you could easily have a subtle
         error, and not even know it.

        Deeplearning4j comes with a gradient check utility that you can use to check your layers.
        This utility works for feed-forward layers, CNNs, RNNs etc.
        For more details on gradient checks, and some references, see the Javadoc for the GradientCheckUtil class:
        https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/gradientcheck/GradientCheckUtil.java

        There are a few things to note when doing gradient checks:
        1. It is necessary to use double precision for ND4J. Single precision (float - the default) isn't sufficiently
           accurate for reliably performing gradient checks
        2. It is necessary to set the updater to None, or equivalently use both the SGD updater and learning rate of 1.0
           Reason: we are testing the raw gradients before they have been modified with learning rate, momentum, etc.
        */

        Span span =
             tracer.spanBuilder("doGradientCheck")
                 .setStartTimestamp(Instant.now())
                 .setAttribute("thread.id", Thread.currentThread().getId())
                 // The above is a semantic attribute
                 .startSpan();

        try (Scope scope = span.makeCurrent()) {
            System.out.println("\n\n\n----- Starting Gradient Check -----");

            Nd4j.getRandom().setSeed(12345);
            int nIn = 3;
            int nOut = 2;

            MultiLayerConfiguration config =
                new NeuralNetConfiguration.Builder().dataType(DataType.DOUBLE)
                    .seed(12345)
                    .updater(new NoOp())
                    .weightInit(new NormalDistribution(0, 1))
                    .l2(0.03)
                    .list()
                    .layer(0, new DenseLayer.Builder()
                                  .activation(Activation.TANH)
                                  .nIn(nIn)
                                  .nOut(3)
                                  .build())
                    .layer(1, new CustomLayer.Builder()
                                  .activation(Activation.TANH)
                                  .secondActivationFunction(Activation.SIGMOID)
                                  .nIn(3).nOut(3)
                                  .build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                  .activation(Activation.SOFTMAX)
                                  .nIn(3).nOut(nOut)
                                  .build())
                    .build();
            MultiLayerNetwork net = new MultiLayerNetwork(config);
            net.init();

            boolean print = true;                      //Whether to print status for each parameter during testing
            boolean return_on_first_failure = false;   //If true: terminate test on first failure
            double gradient_check_epsilon = 1e-8;      //Epsilon value used for gradient checks
            double max_relative_error = 1e-5;          //Maximum relative error allowable for each parameter
            double min_absolute_error = 1e-10;         //Minimum absolute error, to avoid failures on 0 vs 1e-30, for example.

            //Create some random input data to use in the gradient check
            int minibatchSize = 5;
            INDArray features = Nd4j.rand(minibatchSize, nIn);
            INDArray labels = Nd4j.zeros(minibatchSize, nOut);
            Random r = new Random(12345);
            for (int i = 0; i < minibatchSize; i++) {
                labels.putScalar(i, r.nextInt(nOut), 1);
            }

            // Print the number of parameters in each layer. This can help to
            // identify the layer that any failing parameters belong to.
            for (int i = 0; i < 3; i++) {
                System.out.println("# params, layer " + i + ":\t"
                                   + net.getLayer(i).numParams());
            }

            GradientCheckUtil.MLNConfig mlnConfig =
                        new GradientCheckUtil.MLNConfig()
                            .net(net)
                            .epsilon(gradient_check_epsilon)
                            .maxRelError(max_relative_error)
                            .minAbsoluteError(min_absolute_error)
                            .print(GradientCheckUtil.PrintMode.FAILURES_ONLY)
                            .exitOnFirstError(return_on_first_failure)
                            .input(features)
                            .labels(labels);

            GradientCheckUtil.checkGradients(mlnConfig);
        } finally {
            span.end();
        }
    }


    public static double runMnistTest(final Activation firstActivation,
                                      final Activation secondActivation)
        throws IOException {

        final String custLayerName = "myCustomLayer";
        final String convLayerName = "ConvLayer";

        Span span =
             tracer.spanBuilder("runMnistTest")
                 .setStartTimestamp(Instant.now())
                 .setAttribute("firstActivation", firstActivation.toString())
                 .setAttribute("secondActivation", secondActivation.toString())
                 .setAttribute("thread.id", Thread.currentThread().getId())
                 // The above is a semantic attribute
                 .startSpan();

        double accuracy;

        try {
            int nChannels = 1; // Number of input channels
            int nEpochs = 1; // Number of training epochs
            int seed = 123; //

            mnistTrain.reset();
            mnistTest.reset();

            /*
                Construct the neural network
             */
            System.out.printf("---- Building MNIST model for activations: %s %s\n",
                              firstActivation, secondActivation);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .l2(0.0005)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(1e-3))
                    .list()
                    .layer(new ConvolutionLayer.Builder(5, 5)
                            // nIn and nOut specify depth. nIn here is the
                            // nChannels and nOut is the number of filters to
                            // be applied
                            .nIn(nChannels)
                            .stride(1, 1)
                            .nOut(20)
                            .activation(Activation.IDENTITY)
                            .build())
                    .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                    .layer(new ConvolutionLayer.Builder(5, 5)
                             .name(convLayerName)
                            // nIn need not be specified in later layers
                            .stride(1, 1)
                            .nOut(50)
                            .activation(Activation.IDENTITY)
                            .build())
                    .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())

                    /*
                    .layer(new DenseLayer.Builder().activation(Activation.RELU)
                            .nOut(500).build())
                     */

                    .layer(new CustomLayer.Builder()
                             .name(custLayerName)
                             .activation(firstActivation)
                             .secondActivationFunction(secondActivation)
                             .nOut(500)
                             .build())

                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                            .nOut(outputNum)
                            .activation(Activation.SOFTMAX)
                            .build())
                    // See note below
                    .setInputType(InputType.convolutionalFlat(28, 28, 1))
                    .build();

            /*
            Regarding the .setInputType(InputType.convolutionalFlat(28, 28, 1)) line: This does a few things.
            (a) It adds preprocessors, which handle things like the transition between the convolutional/subsampling layers
                and the dense layer
            (b) Does some additional configuration validation
            (c) Where necessary, sets the nIn (number of input neurons, or input depth in the case of CNNs) values for each
                layer based on the size of the previous layer (but it won't override values manually set by the user)

            InputTypes can be used with other layer types too (RNNs, MLPs etc) not just CNNs.
            For normal images (when using ImageRecordReader) use InputType.convolutional(height,width,depth).
            MNIST record reader is a special case, that outputs 28x28 pixel grayscale (nChannels=1) images, in a "flattened"
            row vector format (i.e., 1x784 vectors), hence the "convolutionalFlat" input type used here.
            */

            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

            // Set custom layer's OpenTelemetry tracer:
            Layer custLayer = model.getLayer(custLayerName);
            System.out.printf("DEBUG: Class name: %s\n",
                              custLayer.getClass().getName());
            assert (custLayer instanceof layer.CustomLayerImplementation);
            ((layer.CustomLayerImplementation)
                             custLayer).setOpenTelTracer(tracer);

            System.out.printf("---- Training MNIST model for activations: %s %s\n",
                              firstActivation, secondActivation);

            if (setListenersDuringTraining) {
                // Print score every 10 iterations
                model.setListeners(new ScoreIterationListener(10));
            }

            model.fit(mnistTrain, nEpochs);
            // score = model.score();

            System.out.printf("---- Doing evaluation for activations: %s %s\n",
                              firstActivation, secondActivation);
            mnistTest.reset();
            Evaluation evalTestSet = new Evaluation();
            model.doEvaluation(mnistTest, evalTestSet);
            System.out.printf("%s\n", evalTestSet.stats());
            accuracy = evalTestSet.accuracy();
            System.out.printf("---- Accuracy of %s %s: %f\n",
                              firstActivation, secondActivation, accuracy);
            // It could be a OpenTelemetry span.addEvent(...) to emit the
            // accuracy of the evaluation with these hyper-parameters, but
            // leaving the span mainly for resource-consumption information

            /*
            String path = FilenameUtils.concat(
                              System.getProperty("java.io.tmpdir"),
                              String.format("lenetmnist_%s_%s.zip",
                                            firstActivation, secondActivation)
                          );
            System.out.printf("---- Saving model for activations %s %s to: %s\n",
                              firstActivation, secondActivation, path);
            model.save(new File(path), true);
            */

            System.out.printf("---- MNIST done for activations: %s %s\n",
                              firstActivation, secondActivation);
        } finally {
            span.end();
        }

        return accuracy;
    }
}
