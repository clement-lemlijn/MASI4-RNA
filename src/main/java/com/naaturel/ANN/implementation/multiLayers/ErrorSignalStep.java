package com.naaturel.ANN.implementation.multiLayers;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;

public class ErrorSignalStep implements AlgorithmStep {

    private final GradientBackpropagationContext context;

    public ErrorSignalStep(GradientBackpropagationContext context) {
        this.context = context;
    }

    @Override
    public void run() {

        context.model.forEachNeuron(n -> {
            if (context.errorSignalsComputed[n.getId()]) return;

            int neuronIndex = context.model.indexInLayerOf(n);
            float[] signalSum = {0f};
            context.model.forEachNeuronConnectedTo(n, connected -> {
                signalSum[0] += context.errorSignals[connected.getId()] * connected.getWeight(neuronIndex);
            });

            context.errorSignals[n.getId()] = n.getActivationFunction().derivative(n.getOutput()) * signalSum[0];
            context.errorSignalsComputed[n.getId()] = true;
        });
    }
}