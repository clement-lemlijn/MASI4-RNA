package com.naaturel.ANN.implementation.gradientDescent;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;

import java.util.concurrent.atomic.AtomicInteger;

public class GradientDescentCorrectionStrategy implements AlgorithmStep {

    private final GradientDescentTrainingContext context;

    public GradientDescentCorrectionStrategy(GradientDescentTrainingContext context) {
        this.context = context;
    }

    @Override
    public void run() {
        AtomicInteger i = new AtomicInteger(0);
        context.model.forEachSynapse(syn -> {
            float corrector = context.correctorTerms.get(i.get());
            float c = syn.getWeight() + corrector;
            syn.setWeight(c);
            i.incrementAndGet();
        });
    }
}
