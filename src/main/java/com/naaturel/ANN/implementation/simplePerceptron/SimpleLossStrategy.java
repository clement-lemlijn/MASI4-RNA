package com.naaturel.ANN.implementation.simplePerceptron;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;

public class SimpleLossStrategy implements AlgorithmStep {

    private final SimpleTrainingContext context;

    public SimpleLossStrategy(SimpleTrainingContext context) {
        this.context = context;
    }

    @Override
    public void run() {
        float loss = 0f;
        for (float d : context.deltas) {
            loss += d;
        }
        context.localLoss = loss;
    }
}
