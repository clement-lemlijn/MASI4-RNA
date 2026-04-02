package com.naaturel.ANN.implementation.simplePerceptron;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.domain.abstraction.TrainingContext;
import com.naaturel.ANN.domain.model.neuron.Input;

import java.util.List;

public class SimplePredictionStep implements AlgorithmStep {

    private final TrainingContext context;

    public SimplePredictionStep(TrainingContext context) {
        this.context = context;
    }

    @Override
    public void run() {
        List<Input> data = context.currentEntry.getData();
        float[] flatData = new float[data.size()];
        for (int i = 0; i < data.size(); i++) {
            flatData[i] = data.get(i).getValue();
        }
        context.predictions = context.model.predict(flatData);
    }
}
