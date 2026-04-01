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
        Stream<Float> deltaStream = this.context.deltas.stream();
        this.context.localLoss = deltaStream.reduce(0.0F, (acc, d) -> (float) (acc + Math.pow(d, 2)));
        this.context.localLoss /= 2;
        this.context.globalLoss += this.context.localLoss; //broke MSE en gradientDescentTraining
    }
}
