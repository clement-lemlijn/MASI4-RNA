package com.naaturel.ANN.implementation.multiLayers;

import com.naaturel.ANN.domain.abstraction.ActivationFunction;
import com.naaturel.ANN.domain.model.neuron.Neuron;

public class Sigmoid implements ActivationFunction {

    private float steepness;

    public Sigmoid(float steepness) {
        this.steepness = steepness;
    }

    @Override
    public float accept(Neuron n) {
        return (float) (1.0/(1.0 + Math.exp(-steepness * n.calculateWeightedSum())));
    }

    @Override
    public float derivative(float value) {
        return steepness * value * (1 - value);
    }
}
