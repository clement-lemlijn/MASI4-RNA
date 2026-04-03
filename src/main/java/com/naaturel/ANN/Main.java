package com.naaturel.ANN;

import com.naaturel.ANN.domain.abstraction.Model;
import com.naaturel.ANN.domain.abstraction.Network;
import com.naaturel.ANN.domain.model.neuron.Neuron;
import com.naaturel.ANN.domain.abstraction.Trainer;
import com.naaturel.ANN.implementation.gradientDescent.Linear;
import com.naaturel.ANN.implementation.multiLayers.Sigmoid;
import com.naaturel.ANN.implementation.multiLayers.TanH;
import com.naaturel.ANN.implementation.training.GradientBackpropagationTraining;
import com.naaturel.ANN.infrastructure.config.ConfigDto;
import com.naaturel.ANN.infrastructure.config.ConfigLoader;
import com.naaturel.ANN.infrastructure.dataset.DataSet;
import com.naaturel.ANN.infrastructure.dataset.DataSetEntry;
import com.naaturel.ANN.infrastructure.dataset.DatasetExtractor;
import com.naaturel.ANN.domain.model.neuron.*;
import com.naaturel.ANN.infrastructure.graph.GraphVisualizer;
import com.naaturel.ANN.infrastructure.persistence.ModelSnapshot;

import java.io.Console;
import java.util.*;

public class Main {

    public static void main(String[] args) throws Exception {

        ConfigDto config = ConfigLoader.load("C:/Users/Laurent/Desktop/ANN-framework/config.json");

        boolean newModel = config.getModelProperty("new", Boolean.class);
        int[] modelParameters = config.getModelProperty("parameters", int[].class);
        String modelPath = config.getModelProperty("path", String.class);
        int maxEpoch = config.getTrainingProperty("max_epoch", Integer.class);
        float learningRate = config.getTrainingProperty("learning_rate", Double.class).floatValue();
        String datasetPath = config.getDatasetProperty("path", String.class);

        int nbrClass = 1;
        DataSet dataset = new DatasetExtractor().extract(datasetPath, nbrClass);
        int nbrInput = dataset.getNbrInputs();


        ModelSnapshot snapshot;

        Model network;
        if(newModel){
            network = createNetwork(modelParameters, nbrInput);
            snapshot = new ModelSnapshot(network);
            System.out.println("Parameters: " + network.synCount());
            Trainer trainer = new GradientBackpropagationTraining();
            trainer.train(learningRate, maxEpoch, network, dataset);
        } else {
            snapshot = new ModelSnapshot();
            snapshot.loadFromFile(modelPath);
            network = snapshot.getModel();
        }

        plotGraph(dataset, network);
        snapshot.saveToFile(modelPath);
    }

    private static FullyConnectedNetwork createNetwork(int[] neuronPerLayer, int nbrInput){
        int neuronId = 0;
        List<Layer> layers = new ArrayList<>();
        for (int i = 0; i < neuronPerLayer.length; i++){

            List<Neuron> neurons = new ArrayList<>();
            for (int j = 0; j < neuronPerLayer[i]; j++){

                int nbrSyn = i == 0 ? nbrInput: neuronPerLayer[i-1];

                List<Synapse> syns = new ArrayList<>();
                for (int k=0; k < nbrSyn; k++){
                    syns.add(new Synapse(new Input(0), new Weight()));
                }

                Bias bias =  new Bias(new Weight());

                Neuron n = new Neuron(neuronId, syns.toArray(new Synapse[0]), bias, new TanH());
                neurons.add(n);
                neuronId++;
            }
            Layer layer = new Layer(neurons.toArray(new Neuron[0]));
            layers.add(layer);
        }

        return new FullyConnectedNetwork(layers.toArray(new Layer[0]));
    }

    private static void plotGraph(DataSet dataset, Model network){
        GraphVisualizer visualizer = new GraphVisualizer();

        for (DataSetEntry entry : dataset) {
            List<Float> label = dataset.getLabelsAsFloat(entry);
            label.forEach(l -> {
                visualizer.addPoint("Label " + l,
                        entry.getData().get(0).getValue(), entry.getData().get(1).getValue());
            });
        }

        float min = -0F;
        float max = 10F;
        float step = 0.03F;
        for (float x = min; x < max; x+=step){
            for (float y = min; y < max; y+=step){
                float[] predictions = network.predict(new float[]{x, y});
                visualizer.addPoint(Float.toString(Math.round(predictions[0])), x, y);
            }
        }

        visualizer.buildScatterGraph((int)min-1, (int)max+1);
    }

}
