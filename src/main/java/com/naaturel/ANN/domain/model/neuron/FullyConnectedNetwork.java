package com.naaturel.ANN.domain.model.neuron;

import com.naaturel.ANN.domain.abstraction.Model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * Represents a fully connected neural network
 */
public class FullyConnectedNetwork implements Model {

    private final Layer[] layers;
    private final Map<Neuron, List<Neuron>> connectionMap;
    private final Map<Neuron, Integer> layerIndexByNeuron;
    public FullyConnectedNetwork(Layer[] layers) {
        this.layers = layers;
        this.connectionMap = this.createConnectionMap();
        this.layerIndexByNeuron = this.createNeuronIndex();
    }

    @Override
    public List<Float> predict(List<Input> inputs) {
        List<Input> previousLayerOutputs = new ArrayList<>(inputs);
        for(Layer layer : this.layers){
            List<Float> currentLayerOutputs = layer.predict(previousLayerOutputs);
            previousLayerOutputs = currentLayerOutputs.stream().map(Input::new).toList();
        }
        return previousLayerOutputs.stream().map(Input::getValue).toList();
    }

    @Override
    public int synCount() {
        int res = 0;
        for(Layer layer : this.layers){
            res += layer.synCount();
        }
        return res;
    }

    @Override
    public int neuronCount() {
        int res = 0;
        for(Layer layer : this.layers){
            res += layer.neuronCount();
        }
        return res;
    }

    @Override
    public void forEachSynapse(Consumer<Synapse> consumer) {
        for(Layer l : this.layers){
            l.forEachSynapse(consumer);
        }
    }

    @Override
    public void forEachNeuron(Consumer<Neuron> consumer) {
        for(Layer l : this.layers){
            l.forEachNeuron(consumer);
        }
    }

    @Override
    public void forEachOutputNeurons(Consumer<Neuron> consumer) {
        int lastIndex = this.layers.length-1;
        this.layers[lastIndex].forEachNeuron(consumer);
    }

    @Override
    public void forEachNeuronConnectedTo(Neuron n, Consumer<Neuron> consumer) {
        this.connectionMap.get(n).forEach(consumer);
    }

    @Override
    public int indexInLayerOf(Neuron n) {
        int layerIndex = this.layerIndexByNeuron.get(n);
        return this.layers[layerIndex].indexInLayerOf(n);
    }

    private Map<Neuron, List<Neuron>> createConnectionMap() {
        Map<Neuron, List<Neuron>> res = new HashMap<>();

        for (int i = 0; i < this.layers.length - 1; i++) {
            List<Neuron> nextLayerNeurons = new ArrayList<>();
            this.layers[i + 1].forEachNeuron(nextLayerNeurons::add);
            this.layers[i].forEachNeuron(n -> res.put(n, nextLayerNeurons));
        }
        return res;
    }

    private Map<Neuron, Integer> createNeuronIndex() {
        Map<Neuron, Integer> res = new HashMap<>();
        AtomicInteger index = new AtomicInteger(0);
        for(Layer l : this.layers){
            l.forEachNeuron(n -> res.put(n, index.get()));
            index.incrementAndGet();
        }
        return res;
    }
}
