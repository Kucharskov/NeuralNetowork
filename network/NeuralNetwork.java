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

    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    Matrix weights_ih;
    Matrix weights_ho;
    Matrix bias_h;
    Matrix bias_o;
    double learning_rate = 0.1;

    public NeuralNetwork(int in_nodes, int hid_nodes, int out_nodes) {
        this.input_nodes = in_nodes;
        this.hidden_nodes = hid_nodes;
        this.output_nodes = out_nodes;

        this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
        this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
        this.weights_ih.randomize();
        this.weights_ho.randomize();

        this.bias_h = new Matrix(this.hidden_nodes, 1);
        this.bias_o = new Matrix(this.output_nodes, 1);
        this.bias_h.randomize();
        this.bias_o.randomize();
    }

    public Matrix predict(double[] input_array) {

        // Generating the Hidden Outputs
        Matrix inputs = Matrix.fromArray(input_array);
        Matrix hidden = Matrix.mult(weights_ih, inputs).add(bias_h).map(new Sigmoid.Func());

        // Generating the output's output!
        Matrix outputs = Matrix.mult(weights_ho, hidden).add(bias_o).map(new Sigmoid.Func());

        // Sending back to the caller!
        return outputs;
    }

    public void train(double[] input_array, double[] target_array) {
        // Generating the Hidden Outputs
        Matrix inputs = Matrix.fromArray(input_array);
        Matrix hidden = Matrix.mult(weights_ih, inputs).add(bias_h).map(new Sigmoid.Func());

        // Generating the output's output!
        Matrix outputs = Matrix.mult(weights_ho, hidden).add(bias_o).map(new Sigmoid.Func());

        // Convert array to matrix object
        Matrix targets = Matrix.fromArray(target_array);

        // Calculate the error
        // ERROR = TARGETS - OUTPUTS
        Matrix output_errors = Matrix.sub(targets, outputs);

        // let gradient = outputs * (1 - outputs);
        // Calculate gradient
        Matrix gradients = new Matrix(outputs).map(new Sigmoid.Dfunc()).multHadamar(output_errors).mult(learning_rate);

        // Calculate deltas
        Matrix hidden_T = Matrix.transpose(hidden);
        Matrix weight_ho_deltas = Matrix.mult(gradients, hidden_T);

        // Adjust the weights by deltas
        weights_ho.add(weight_ho_deltas);
        // Adjust the bias by its deltas (which is just the gradients)
        bias_o.add(gradients);

        // Calculate the hidden layer errors
        Matrix hidden_errors = Matrix.mult(Matrix.transpose(weights_ho), output_errors);

        // Calculate hidden gradient
        Matrix hidden_gradient = new Matrix(hidden).map(new Sigmoid.Dfunc()).multHadamar(hidden_errors).mult(learning_rate);

        // Calcuate input->hidden deltas
        Matrix weight_ih_deltas = Matrix.mult(hidden_gradient, Matrix.transpose(inputs));

        weights_ih.add(weight_ih_deltas);
        // Adjust the bias by its deltas (which is just the gradients)
        bias_h.add(hidden_gradient);
    }
}
