package com.naaturel.ANN.implementation.simplePerceptron;

import com.naaturel.ANN.domain.abstraction.ActivationFunction;
import com.naaturel.ANN.domain.model.neuron.Neuron;

import javax.naming.OperationNotSupportedException;

public class Heaviside implements ActivationFunction {

    public Heaviside(){

    }

    @Override
    public float accept(Neuron n) {
        float weightedSum = n.calculateWeightedSum();
        return weightedSum < 0 ? 0:1;
    }

    @Override
    public float derivative(float value) {
        throw new UnsupportedOperationException("Heaviside is not differentiable");
    }
}
