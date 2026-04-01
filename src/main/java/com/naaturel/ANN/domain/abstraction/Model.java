package com.naaturel.ANN.domain.abstraction;

import com.naaturel.ANN.domain.model.neuron.Input;
import com.naaturel.ANN.domain.model.neuron.Neuron;
import com.naaturel.ANN.domain.model.neuron.Synapse;

import java.util.List;
import java.util.function.Consumer;

public interface Model {
    int synCount();
    int neuronCount();
    int indexInLayerOf(Neuron n);
    void forEachNeuron(Consumer<Neuron> consumer);
    void forEachSynapse(Consumer<Synapse> consumer);
    void forEachOutputNeurons(Consumer<Neuron> consumer);
    void forEachNeuronConnectedTo(Neuron n, Consumer<Neuron> consumer);
    List<Float> predict(List<Input> inputs);
}
