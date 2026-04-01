package com.naaturel.ANN.implementation.simplePerceptron;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.domain.abstraction.TrainingContext;

import java.util.concurrent.atomic.AtomicInteger;


public class SimpleCorrectionStep implements AlgorithmStep {

    private final TrainingContext context;

    public SimpleCorrectionStep(TrainingContext context) {
        this.context = context;
    }

    @Override
    public void run() {
        if(context.expectations.equals(context.predictions)) return;
        AtomicInteger neuronIndex = new AtomicInteger(0);
        AtomicInteger synIndex = new AtomicInteger(0);

        context.model.forEachNeuron(neuron -> {
            float correspondingDelta = context.deltas.get(neuronIndex.get());
            neuron.forEachSynapse(syn -> {
                float currentW = syn.getWeight();
                float currentInput  = syn.getInput();
                float newValue = currentW + (context.learningRate * correspondingDelta * currentInput);
                syn.setWeight(newValue);
                synIndex.incrementAndGet();
            });
            neuronIndex.incrementAndGet();
        });
    }
}
