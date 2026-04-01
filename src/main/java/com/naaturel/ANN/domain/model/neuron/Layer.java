package com.naaturel.ANN.domain.model.neuron;

import com.naaturel.ANN.domain.abstraction.Model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public class Layer implements Model {

    private final Neuron[] neurons;
    private final Map<Neuron, Integer> neuronIndex;

    public Layer(Neuron[] neurons) {
        this.neurons = neurons;
        this.neuronIndex = createNeuronIndex();
    }

    @Override
    public List<Float> predict(List<Input> inputs) {
        List<Float> result = new ArrayList<>();
        for(Neuron neuron : this.neurons){
            List<Float> res = neuron.predict(inputs);
            result.addAll(res);
        }
        return result;
    }

    @Override
    public int synCount() {
        int res = 0;
        for (Neuron neuron : this.neurons) {
            res += neuron.synCount();
        }
        return res;
    }

    @Override
    public int neuronCount() {
        return this.neurons.length;
    }

    @Override
    public int indexInLayerOf(Neuron n) {
        return this.neuronIndex.get(n);
    }

    @Override
    public void forEachNeuron(Consumer<Neuron> consumer) {
        for (Neuron n : this.neurons){
            consumer.accept(n);
        }
    }

    @Override
    public void forEachSynapse(Consumer<Synapse> consumer) {
        for (Neuron n : this.neurons){
            n.forEachSynapse(consumer);
        }
    }

    @Override
    public void forEachOutputNeurons(Consumer<Neuron> consumer) {
        this.forEachNeuron(consumer);
    }

    @Override
    public void forEachNeuronConnectedTo(Neuron n, Consumer<Neuron> consumer) {
        throw new UnsupportedOperationException("Neurons have no connection within the same layer");
    }

    private Map<Neuron, Integer> createNeuronIndex() {
        Map<Neuron, Integer> res = new HashMap<>();
        int[] index = {0};
        this.forEachNeuron(n -> {
            res.put(n, index[0]++);
        });
        return res;
    }

}
