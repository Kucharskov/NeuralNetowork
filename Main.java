/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn;

import java.util.Random;
import nn.network.NeuralNetwork;

/**
 *
 * @author M.Kucharskov
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(2, 1);
        nn.addHiddenLayer(2);
		
        for (int i = 0; i < 50000; i++) {
            switch (new Random().nextInt(4)) {
                case 0:
                    nn.train(new double[]{0, 0}, new double[]{0});
                    break;
                case 1:
                    nn.train(new double[]{0, 1}, new double[]{1});
                    break;
                case 2:
                    nn.train(new double[]{1, 0}, new double[]{1});
                    break;
                case 3:
                    nn.train(new double[]{1, 1}, new double[]{0});
                    break;
            }
        }
        
        System.out.println("XOR");
        System.out.println("0, 0: " + nn.predict(new double[]{0,0})[0]);
        System.out.println("0, 1: " + nn.predict(new double[]{0,1})[0]);
        System.out.println("1, 0: " + nn.predict(new double[]{1,0})[0]);
        System.out.println("1, 1: " + nn.predict(new double[]{1,1})[0]);
    }
}
