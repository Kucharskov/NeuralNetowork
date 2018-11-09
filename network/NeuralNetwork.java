/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.network;

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
    Matrix weightsInputHidden;
    Matrix weightsHiddenOutput;
    Matrix biasHidden;
    Matrix biasOutput;
    double learningRate;

    public NeuralNetwork(int input, int hidden, int output) {
        this.inputNodes = input;
        this.hiddenNodes = hidden;
        this.outputNodes = output;

        this.weightsInputHidden = new Matrix(this.hiddenNodes, this.inputNodes);
        this.weightsHiddenOutput = new Matrix(this.outputNodes, this.hiddenNodes);
        this.weightsInputHidden.randomize();
        this.weightsHiddenOutput.randomize();

        this.biasHidden = new Matrix(this.hiddenNodes, 1);
        this.biasOutput = new Matrix(this.outputNodes, 1);
        this.biasHidden.randomize();
        this.biasOutput.randomize();

        this.learningRate = 0.1;
    }

    public NeuralNetwork(int input, int hidden, int output, int learningRate) {
        this(input, hidden, output);
        this.learningRate = learningRate;
    }

    public double[] predict(double[] inputArray) {
        // Generating the Hidden Outputs
        Matrix inputs = Matrix.fromArray(inputArray);
        Matrix hidden = Matrix.mult(weightsInputHidden, inputs)
                .add(biasHidden)
                .map(new Sigmoid.Func());

        // Generating the output's output!
        Matrix outputs = Matrix.mult(weightsHiddenOutput, hidden)
                .add(biasOutput)
                .map(new Sigmoid.Func());

        // Returning outputs
        return outputs.toArray();
    }

    public void train(double[] inputArray, double[] targetArray) {
        // Generating the Hidden Outputs
        Matrix inputs = Matrix.fromArray(inputArray);
        Matrix hidden = Matrix.mult(weightsInputHidden, inputs)
                .add(biasHidden)
                .map(new Sigmoid.Func());

        // Generating the output's output!
        Matrix outputs = Matrix.mult(weightsHiddenOutput, hidden)
                .add(biasOutput)
                .map(new Sigmoid.Func());

        // Convert array to matrix object
        Matrix targets = Matrix.fromArray(targetArray);

        // Calculate the error
        // ERROR = TARGETS - OUTPUTS
        Matrix outputErrors = Matrix.sub(targets, outputs);

        // let gradient = outputs * (1 - outputs);
        // Calculate gradient
        Matrix gradients = new Matrix(outputs)
                .map(new Sigmoid.Dfunc())
                .multHadamar(outputErrors)
                .mult(learningRate);

        // Calculate deltas
        Matrix hiddenTranspose = Matrix.transpose(hidden);
        Matrix weightHiddenOutputDeltas = Matrix.mult(gradients, hiddenTranspose);

        // Adjust the weights by deltas
        weightsHiddenOutput.add(weightHiddenOutputDeltas);
        // Adjust the bias by its deltas (which is just the gradients)
        biasOutput.add(gradients);

        // Calculate the hidden layer errors
        Matrix weightsHiddenOutputTranspose = Matrix.transpose(weightsHiddenOutput);
        Matrix hiddenErrors = Matrix.mult(weightsHiddenOutputTranspose, outputErrors);

        // Calculate hidden gradient
        Matrix hiddenGradient = new Matrix(hidden)
                .map(new Sigmoid.Dfunc())
                .multHadamar(hiddenErrors)
                .mult(learningRate);

        // Calcuate input->hidden deltas
        Matrix inputTranspose = Matrix.transpose(inputs);
        Matrix weightInputHiddenDeltas = Matrix.mult(hiddenGradient, inputTranspose);

        weightsInputHidden.add(weightInputHiddenDeltas);
        // Adjust the bias by its deltas (which is just the gradients)
        biasHidden.add(hiddenGradient);
    }
}
