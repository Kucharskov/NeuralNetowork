/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn.matrix;

import nn.function.CallableFunction;

/**
 *
 * @author M.Kucharskov
 */
public class Matrix {

    int rows;
    int cols;
    double[] array;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;

        this.array = new double[this.rows * this.cols];
        for (int i = 0; i < array.length; i++) {
            array[i] = 0;
        }
    }
    
    public Matrix(Matrix m) {
        this.array = m.array;
        this.rows = m.rows;
        this.cols = m.cols;
    }

    //Getter rzędów
    public int getRows() {
        return rows;
    }

    //Getter kolumn
    public int getCols() {
        return cols;
    }

    //Zabezpieczone pobieranie elementu z macierzy
    public double getElement(int row, int col) {
        if (row < 0 || col < 0 || row * col >= array.length) {
            throw new ArrayIndexOutOfBoundsException("Wrong element coordinates");
        }
        return array[getIndex(row, col)];
    }

    //Inicializacja losowymi liczbami z zakresu <-1, 1>
    public Matrix randomize() {
        for (int i = 0; i < array.length; i++) {
            array[i] = Math.random() * 2 - 1;
        }
        return this;
    }

    //Dodawanie liczby do macierzy
    public Matrix add(double n) {
        for (int i = 0; i < array.length; i++) {
            array[i] += n;
        }
        return this;
    }

    //Odejmowanie liczby do macierzy
    public Matrix sub(double n) {
        for (int i = 0; i < array.length; i++) {
            array[i] -= n;
        }
        return this;
    }

    //Mnożenie macierzy przez liczbę
    public Matrix mult(double n) {
        for (int i = 0; i < array.length; i++) {
            array[i] *= n;
        }
        return this;
    }
    
    public Matrix map(CallableFunction af) {
        for (int i = 0; i < array.length; i++) {
            array[i] = af.call(array[i]);
        }
        return this;
    }

    //Dodawanie macierzy do macierzy
    public Matrix add(Matrix m) {
        Matrix result = Matrix.add(this, m);

        this.array = result.array;
        this.rows = result.rows;
        this.cols = result.cols;
        
        return this;
    }

    //Odejmowanie macierzy do macierzy
    public Matrix sub(Matrix m) {
        Matrix result = Matrix.sub(this, m);

        this.array = result.array;
        this.rows = result.rows;
        this.cols = result.cols;
        
        return this;
    }

    //Mnożenie macierzy przez macierz
    public Matrix mult(Matrix m) {
        Matrix result = Matrix.mult(this, m);

        this.array = result.array;
        this.rows = result.rows;
        this.cols = result.cols;
        
        return this;
    }
   
    //Mnożenie Hadamara macierzy przez macierz
    public Matrix multHadamar(Matrix m) {
        Matrix result = Matrix.multHadamar(this, m);

        this.array = result.array;
        this.rows = result.rows;
        this.cols = result.cols;
        
        return this;
    }

    //Transpozycja macierzy
    public Matrix transpose() {
        Matrix result = Matrix.transpose(this);

        this.array = result.array;
        this.rows = result.rows;
        this.cols = result.cols;
        
        return this;
    }

    //Statyczne dodawanie macierzy do macierzy
    public static Matrix add(Matrix m1, Matrix m2) {
        if (m1.rows != m2.rows || m1.cols != m2.cols) {
            throw new IllegalArgumentException("Wrong add Matrix dimensions");
        }

        Matrix result = new Matrix(m1.rows, m1.cols);
        for (int i = 0; i < result.array.length; i++) {
            result.array[i] = m1.array[i] + m2.array[i];
        }
        return result;
    }

    //Statyczne odejmowanie macierzy do macierzy
    public static Matrix sub(Matrix m1, Matrix m2) {
        if (m1.rows != m2.rows || m1.cols != m2.cols) {
            throw new IllegalArgumentException("Wrong sub Matrix dimensions");
        }

        Matrix result = new Matrix(m1.rows, m1.cols);
        for (int i = 0; i < result.array.length; i++) {
            result.array[i] = m1.array[i] - m2.array[i];
        }
        return result;
    }

    //Statyczne mnożenie macierzy przez macierz
    public static Matrix mult(Matrix m1, Matrix m2) {
        if (m1.cols != m2.rows) {
            throw new IllegalArgumentException("Wrong multiply Matrix dimensions");
        }

        Matrix result = new Matrix(m1.rows, m2.cols);
        for (int i = 0; i < m1.rows; i++) {
            for (int j = 0; j < m2.cols; j++) {
                double sum = 0;
                for (int k = 0; k < m1.cols; k++) {
                    sum += m1.getElement(i, k) * m2.getElement(k, j);
                }
                result.array[result.getIndex(i, j)] = sum;
            }
        }

        return result;
    }

    //Statyczne mnożenie Hadamara macierzy przez macierz
    public static Matrix multHadamar(Matrix m1, Matrix m2) {
        if (m1.rows != m2.rows || m1.cols != m2.cols) {
            throw new IllegalArgumentException("Wrong Hadamar multiply Matrix dimensions");
        }

        Matrix result = new Matrix(m1.rows, m1.cols);
        for (int i = 0; i < m1.array.length; i++) {
            result.array[i] = m1.array[i] * m2.array[i];
        }

        return result;
    }

    //Statyczna transpozycja macierzy
    public static Matrix transpose(Matrix m) {
        Matrix result = new Matrix(m.cols, m.rows);
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                result.array[result.getIndex(j, i)] = m.getElement(i, j);
            }
        }

        return result;
    }

    //Metoda statyczna do tworzenia macierzy (wektora) z tablicy
    public static Matrix fromArray(double[] array) {
        Matrix result = new Matrix(array.length, 1);
        result.array = array;

        return result;
    }

    //Prywatna metoda przeliczająca 2D na 1D
    private int getIndex(int row, int col) {
        return (row * cols) + col;
    }

    //Metoda testowa do wyświetlania macierzy
    public void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.print(array[getIndex(i, j)] + " ");
            }
            System.out.println("");
        }
    }
}
