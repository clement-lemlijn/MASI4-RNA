package com.naaturel.ANN.implementation.multiLayers;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.infrastructure.dataset.DataSetEntry;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class OutputLayerErrorStep implements AlgorithmStep {

    private final GradientBackpropagationContext context;

    public OutputLayerErrorStep(GradientBackpropagationContext context){
        this.context = context;
    }

    @Override
    public void run() {
        context.deltas = new ArrayList<>();
        DataSetEntry entry = this.context.currentEntry;
        List<Float> expectations = this.context.dataset.getLabelsAsFloat(entry);
        AtomicInteger index = new AtomicInteger(0);

        context.errorSignals.clear();
        this.context.model.forEachOutputNeurons(n -> {
            float expected = expectations.get(index.get());
            float predicted = n.getOutput();
            float output = n.getOutput();
            float delta = expected - predicted;
            float signal = delta * n.getActivationFunction().derivative(output);

            this.context.deltas.add(delta);
            this.context.errorSignals.put(n, signal);
            index.incrementAndGet();
        });
    }
}
