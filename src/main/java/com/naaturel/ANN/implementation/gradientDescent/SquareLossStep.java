package com.naaturel.ANN.implementation.gradientDescent;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.domain.abstraction.TrainingContext;

import java.util.stream.Stream;

public class SquareLossStep implements AlgorithmStep {

    private final TrainingContext context;

    public SquareLossStep(TrainingContext context) {
        this.context = context;
    }

    @Override
    public void run() {
        float loss = 0f;
        for (float d : this.context.deltas) {
            loss += d * d;
        }
        this.context.localLoss = loss / 2f;
        this.context.globalLoss += this.context.localLoss;
    }
}
