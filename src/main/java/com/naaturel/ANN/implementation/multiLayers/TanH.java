package com.naaturel.ANN.implementation.multiLayers;

import com.naaturel.ANN.domain.abstraction.ActivationFunction;
import com.naaturel.ANN.domain.model.neuron.Neuron;

public class TanH implements ActivationFunction {

    @Override
    public float accept(Neuron n) {
        //For educational purpose. Math.tanh() could have been used here
        float weightedSum = n.calculateWeightedSum();
        double exp = Math.exp(weightedSum);
        double res = (exp-(1/exp))/(exp+(1/exp));
        return (float)(res);
    }

    @Override
    public float derivative(float value) {
        return 1 - value * value;
    }
}
