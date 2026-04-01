package com.naaturel.ANN.domain.abstraction;

import com.naaturel.ANN.domain.model.neuron.Neuron;

public interface ActivationFunction {

    float accept(Neuron n);
    float derivative(float value);

}
