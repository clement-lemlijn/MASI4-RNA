package com.naaturel.ANN.implementation.multiLayers;

import com.naaturel.ANN.domain.abstraction.AlgorithmStep;
import com.naaturel.ANN.domain.model.neuron.Neuron;

import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

public class ErrorSignalStep implements AlgorithmStep {

    private GradientBackpropagationContext context;
    public ErrorSignalStep(GradientBackpropagationContext context) {
        this.context = context;
    }

    @Override
    public void run() {
        this.context.model.forEachNeuron(n -> {
            calculateErrorSignalRecursive(n, this.context.errorSignals);
        });
    }

    private float calculateErrorSignalRecursive(Neuron n, Map<Neuron, Float> signals) {
        if (signals.containsKey(n)) return signals.get(n);

        int neuronIndex =  this.context.model.indexInLayerOf(n);
        AtomicReference<Float> signalSum = new AtomicReference<>(0F);
        this.context.model.forEachNeuronConnectedTo(n, connected -> {
            float weightedSignal = calculateErrorSignalRecursive(connected, signals) * connected.getWeight(neuronIndex);
            signalSum.set(signalSum.get() + weightedSignal);
        });

        float derivative = n.getActivationFunction().derivative(n.getOutput());
        float finalSignal = derivative * signalSum.get();
        signals.put(n, finalSignal);
        return finalSignal;
    }
}
