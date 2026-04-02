package com.naaturel.ANN.implementation.multiLayers;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.infrastructure.dataset.DataSetEntry;

import java.util.Arrays;
import java.util.List;

public class OutputLayerErrorStep implements AlgorithmStep {

    private final GradientBackpropagationContext context;
    private final float[] expectations;

    public OutputLayerErrorStep(GradientBackpropagationContext context){
        this.context = context;
        this.expectations = new float[context.dataset.getNbrLabels()];
    }

    @Override
    public void run() {
        Arrays.fill(context.errorSignals, 0f);
        Arrays.fill(context.errorSignalsComputed, false);

        DataSetEntry entry = context.currentEntry;
        List<Float> labels = context.dataset.getLabelsAsFloat(entry);
        for (int i = 0; i < labels.size(); i++) {
            expectations[i] = labels.get(i);
        }

        int[] index = {0};
        context.model.forEachOutputNeurons(n -> {
            float expected = expectations[index[0]];
            float predicted = n.getOutput();
            float delta = expected - predicted;

            context.deltas[index[0]] = delta;
            context.errorSignals[n.getId()] = delta * n.getActivationFunction().derivative(predicted);
            context.errorSignalsComputed[n.getId()] = true;
            index[0]++;
        });
    }
}