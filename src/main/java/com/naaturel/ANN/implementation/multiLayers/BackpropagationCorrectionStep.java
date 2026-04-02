package com.naaturel.ANN.implementation.multiLayers;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;

public class BackpropagationCorrectionStep implements AlgorithmStep {

    private final GradientBackpropagationContext context;
    private final int synCount;
    private final float[] inputs;
    private final float[] signals;

    public BackpropagationCorrectionStep(GradientBackpropagationContext context){
        this.context = context;
        this.synCount = context.correctionBuffer.length;
        this.inputs = new float[synCount];
        this.signals = new float[synCount];
    }

    @Override
    public void run() {
        int[] synIndex = {0};
        context.model.forEachNeuron(n -> {
            float signal = context.errorSignals[n.getId()];
            for (int i = 0; i < n.synCount(); i++){
                inputs[synIndex[0]] = n.getInput(i);
                signals[synIndex[0]] = signal;
                synIndex[0]++;
            }
        });

        float lr = context.learningRate;
        boolean applyUpdate = context.currentSample >= context.batchSize;

        for (int i = 0; i < synCount; i++) {
            context.correctionBuffer[i] += lr * signals[i] * inputs[i];
        }

        if (applyUpdate) {
            syncWeights();
            context.currentSample = 0;
        }

        context.currentSample++;
    }

    private void syncWeights() {
        int[] synIndex = {0};
        context.model.forEachNeuron(n -> {
            for (int i = 0; i < n.synCount(); i++) {
                n.setWeight(i, n.getWeight(i) + context.correctionBuffer[synIndex[0]]);
                context.correctionBuffer[synIndex[0]] = 0f;
                synIndex[0]++;
            }
        });
    }
}