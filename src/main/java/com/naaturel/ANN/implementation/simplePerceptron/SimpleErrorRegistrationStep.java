package com.naaturel.ANN.implementation.simplePerceptron;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.domain.abstraction.TrainingContext;

public class SimpleErrorRegistrationStep implements AlgorithmStep {

    private final TrainingContext context;

    public SimpleErrorRegistrationStep(TrainingContext context) {
        this.context = context;
    }

    @Override
    public void run() {
        context.globalLoss += context.localLoss;
    }
}
