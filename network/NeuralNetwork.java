/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.network;

import java.util.ArrayList;
import nn.matrix.Matrix;
import nn.function.Sigmoid;

/**
 *
 * @author M.Kucharskov
 */
public class NeuralNetwork {

    int inputNodes;
    int hiddenNodes;
    int outputNodes;
    ArrayList<NeuralNetworkLayer> layers;
    double learningRate;

    public NeuralNetwork(int input, int hidden, int output) {
        if (input <= 0 || hidden <= 0 || output <= 0) {
            throw new IllegalArgumentException("Wrong amount of layer neurons");
        }

        this.inputNodes = input;
        this.hiddenNodes = hidden;
        this.outputNodes = output;

        // Creating the Layers of the Neural Network
        // ! The last Layer is the output layer
        this.layers = new ArrayList<>();
        this.layers.add(new NeuralNetworkLayer(this.inputNodes, this.hiddenNodes));
        // hidden_nodes.length is the last entry at that time
        this.layers.add(new NeuralNetworkLayer(this.hiddenNodes, this.outputNodes));

        this.learningRate = 0.1;
    }

    public NeuralNetwork(int input, int hidden, int output, int learningRate) {
        this(input, hidden, output);
        this.learningRate = learningRate;
    }

    public void addHiddenLayer(int nodes) {
        // Remove old output layer
        layers.remove(layers.size() - 1);
        
        // Add new hidden layer
        int inputNodes = layers.get(layers.size() - 1).outputNodes;
        layers.add(new NeuralNetworkLayer(inputNodes, nodes));
        
        // Add new output layer
        layers.add(new NeuralNetworkLayer(nodes, outputNodes));
    }

    public double[] predict(double[] inputArray) {
        // Generating the Hidden Outputs
        Matrix inputs = Matrix.fromArray(inputArray);

        Matrix prediction = inputs;
        // Loop layer over the inputs
        for (int i = 0; i < layers.size(); i++) {
            prediction = layers.get(i).predict(prediction);
        }
        
        // Returning last prediction
        return prediction.toArray();
    }

    public void train(double[] inputArray, double[] targetArray) {
        // Convert input arrays to matrix objects
        Matrix inputs = Matrix.fromArray(inputArray);
        Matrix targets = Matrix.fromArray(targetArray);

        ArrayList<Matrix> predictions = new ArrayList<>();
        Matrix prediction = inputs;
        // Loop layer over the inputs
        for (int i = 0; i < layers.size(); i++) {
            prediction = layers.get(i).predict(prediction);
            predictions.add(prediction);
        }

        // Last layer == output layer
        Matrix outputs = predictions.get(predictions.size() - 1);

        // Calculate the error
        // ERROR = TARGETS - OUTPUTS
        Matrix currentErrors = Matrix.sub(targets, outputs);
        for (int i = layers.size() - 1; i >= 0; i--) {
            // Calculate deltas
            if (i == 0) {
                currentErrors = layers.get(i).applyError(predictions.get(i), inputs, currentErrors);
            } else {
                currentErrors = layers.get(i).applyError(predictions.get(i), predictions.get(i - 1), currentErrors);
            }
        }
    }

    class NeuralNetworkLayer {

        int inputNodes;
        int outputNodes;
        Matrix weights;
        Matrix bias;

        public NeuralNetworkLayer(int inputNodes, int outputNodes) {
            this.inputNodes = inputNodes;
            this.outputNodes = outputNodes;
            this.weights = new Matrix(outputNodes, inputNodes);
            this.weights.randomize();
            this.bias = new Matrix(outputNodes, 1);
            this.bias.randomize();
        }

        public Matrix predict(Matrix input) {
            Matrix prediction = Matrix.mult(weights, input)
                    .add(bias)
                    .map(new Sigmoid.Func());

            return prediction;
        }

        public Matrix applyError(Matrix prediction, Matrix prevPrediction, Matrix currentErrors) {
            // Calculate the gradients for the layer
            Matrix gradients = new Matrix(prediction)
                    .map(new Sigmoid.Dfunc())
                    .multHadamar(currentErrors)
                    .mult(learningRate);

            // Calculate deltas
            Matrix prevPredictionTranspose = Matrix.transpose(prevPrediction);
            Matrix weightDeltas = Matrix.mult(gradients, prevPredictionTranspose);

            //Apply Errors to the weights and the bias
            weights.add(weightDeltas);
            bias.add(gradients);

            // Calculate the next layer errors
            Matrix weightsTranspose = Matrix.transpose(weights);
            currentErrors = Matrix.mult(weightsTranspose, currentErrors);

            return currentErrors;
        }
    }
}
