package com.naaturel.ANN.implementation.simplePerceptron;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.domain.abstraction.TrainingContext;

public class SimplePredictionStep implements AlgorithmStep {

    private final TrainingContext context;

    public SimplePredictionStep(TrainingContext context) {
        this.context = context;
    }

    @Override
    public void run() {
        context.predictions = context.model.predict(context.currentEntry.getData());
    }
}
