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
        int[] globalSynIndex = {0};
        context.model.forEachNeuron(n -> {
            for(int i = 0; i < n.synCount(); i++){
                float corrector = context.correctorTerms.get(globalSynIndex[0]);
                float c = n.getWeight(i) + corrector;
                n.setWeight(i, c);
                globalSynIndex[0]++;
            }
        });
    }
}
