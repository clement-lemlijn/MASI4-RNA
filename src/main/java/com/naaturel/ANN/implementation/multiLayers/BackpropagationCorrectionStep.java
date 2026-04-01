package com.naaturel.ANN.implementation.multiLayers;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;

import java.util.concurrent.atomic.AtomicInteger;

public class BackpropagationCorrectionStep implements AlgorithmStep {

    private GradientBackpropagationContext context;

    public BackpropagationCorrectionStep(GradientBackpropagationContext context){
        this.context = context;
    }

    @Override
    public void run() {

        AtomicInteger synIndex = new AtomicInteger(0);
        this.context.model.forEachNeuron(n -> {
            float signal = context.errorSignals.get(n);
            n.forEachSynapse(syn -> {
                float lr = context.learningRate;
                float corrector = lr * signal * syn.getInput();
                float existingCorrector = context.correctionBuffer[synIndex.get()];
                float newCorrector = existingCorrector + corrector;

                if(context.currentSample >= context.batchSize){
                    float newWeight = syn.getWeight() + newCorrector;
                    syn.setWeight(newWeight);
                    context.correctionBuffer[synIndex.get()] = 0;
                } else {
                    context.correctionBuffer[synIndex.get()] = newCorrector;
                }
                synIndex.incrementAndGet();
            });
        });
        if(context.currentSample >= context.batchSize) {
            context.currentSample = 0;
        }
        context.currentSample += 1;
    }
}
