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

        context.model.forEachNeuron(neuron -> {
            float correspondingDelta = context.deltas[neuronIndex.get()];

            for(int i = 0; i < neuron.synCount(); i++){
                float currentW = neuron.getWeight(i);
                float currentInput  = neuron.getInput(i);
                float newValue = currentW + (context.learningRate * correspondingDelta * currentInput);
                neuron.setWeight(i, newValue);
            }
            neuronIndex.incrementAndGet();
        });
    }
}
