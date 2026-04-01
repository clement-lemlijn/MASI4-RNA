package com.naaturel.ANN.implementation.simplePerceptron;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;

public class SimpleLossStrategy implements AlgorithmStep {

    private final SimpleTrainingContext context;

    public SimpleLossStrategy(SimpleTrainingContext context) {
        this.context = context;
    }

    @Override
    public void run() {
        this.context.localLoss = this.context.deltas.stream().reduce(0.0F, Float::sum);
    }
}
