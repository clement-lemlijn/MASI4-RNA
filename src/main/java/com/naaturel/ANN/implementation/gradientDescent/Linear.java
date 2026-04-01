package com.naaturel.ANN.implementation.gradientDescent;

import com.naaturel.ANN.domain.abstraction.ActivationFunction;
import com.naaturel.ANN.domain.model.neuron.Neuron;

public class Linear implements ActivationFunction {

    private final float slope;
    private final float intercept;

    public Linear(float slope, float intercept) {
        this.slope = slope;
        this.intercept = intercept;
    }

    @Override
    public float accept(Neuron n) {
        return slope * n.calculateWeightedSum() + intercept;
    }

    @Override
    public float derivative(float value) {
        return this.slope;
    }

}
