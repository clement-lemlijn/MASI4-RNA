package com.naaturel.ANN.domain.model.neuron;

public class Bias extends Synapse {

    public Bias(Weight weight) {
        super(new Input(1), weight);
    }
}
