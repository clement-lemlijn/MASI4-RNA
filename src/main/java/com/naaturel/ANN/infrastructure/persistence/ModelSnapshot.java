package com.naaturel.ANN.infrastructure.persistence;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.naaturel.ANN.domain.abstraction.Model;
import com.naaturel.ANN.domain.model.neuron.*;
import com.naaturel.ANN.implementation.multiLayers.TanH;

import java.io.File;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class ModelSnapshot {

    private Model model;
    private final ObjectMapper mapper;

    public ModelSnapshot(){
        this(null);
    }

    public ModelSnapshot(Model model){
        this.model = model;
        mapper = new ObjectMapper();
    }

    public Model getModel() {
        return model;
    }

    public void saveToFile(String path) throws Exception {

        ArrayNode root = mapper.createArrayNode();
        model.forEachNeuron(n -> {

            ObjectNode neuronNode = mapper.createObjectNode();
            neuronNode.put("id", n.getId());
            neuronNode.put("layerIndex", model.layerIndexOf(n));

            ArrayNode weights = mapper.createArrayNode();
            for (int i = 0; i < n.synCount(); i++) {
                float weight = n.getWeight(i);
                weights.add(weight);
            }
            neuronNode.set("weights", weights);
            root.add(neuronNode);
        });

        mapper.writerWithDefaultPrettyPrinter().writeValue(new File(path), root);
    }

    public void loadFromFile(String path) throws Exception {
        ArrayNode root = (ArrayNode) mapper.readTree(new File(path));

        Map<Integer, List<Neuron>> neuronsByLayer = new LinkedHashMap<>();

        root.forEach(neuronNode -> {
            int id = neuronNode.get("id").asInt();
            int layerIndex = neuronNode.get("layerIndex").asInt();
            ArrayNode weightsNode = (ArrayNode) neuronNode.get("weights");

            Bias bias = new Bias(new Weight(weightsNode.get(0).floatValue()));
            Synapse[] synapses = new Synapse[weightsNode.size() - 1];
            for (int i = 0; i < synapses.length; i++) {
                synapses[i] = new Synapse(new Input(0), new Weight(weightsNode.get(i + 1).floatValue()));
            }

            Neuron n = new Neuron(id, synapses, bias, new TanH());
            neuronsByLayer.computeIfAbsent(layerIndex, k -> new ArrayList<>()).add(n);
        });

        Layer[] layers = neuronsByLayer.values().stream()
                .map(neurons -> new Layer(neurons.toArray(new Neuron[0])))
                .toArray(Layer[]::new);

        this.model = new FullyConnectedNetwork(layers);
    }
}
