package com.naaturel.ANN.implementation.gradientDescent;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;

import java.util.concurrent.atomic.AtomicInteger;

public class GradientDescentErrorStrategy implements AlgorithmStep {

    private final GradientDescentTrainingContext context;

    public GradientDescentErrorStrategy(GradientDescentTrainingContext context) {
        this.context = context;
    }


    @Override
    public void run() {

        AtomicInteger neuronIndex = new AtomicInteger(0);
        AtomicInteger synIndex = new AtomicInteger(0);

        context.model.forEachNeuron(neuron -> {
            float correspondingDelta = context.deltas.get(neuronIndex.get());

            neuron.forEachSynapse(syn -> {
                float corrector = context.correctorTerms.get(synIndex.get());
                corrector += context.learningRate * correspondingDelta * syn.getInput();
                context.correctorTerms.set(synIndex.get(), corrector);
                synIndex.incrementAndGet();
            });

            neuronIndex.incrementAndGet();
        });

        context.globalLoss += context.localLoss;
    }
}
