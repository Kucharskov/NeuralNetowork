/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.function;

/**
 *
 * @author M.Kucharskov
 */
public class Sigmoid {

    public static class Func implements CallableFunction {

        @Override
        public double call(double x) {
            return 1 / (1 + Math.exp(-x));
        }

    }

    public static class Dfunc implements CallableFunction {

        @Override
        public double call(double y) {
            return y * (1 - y);
        }

    }
}
