package com.naaturel.ANN;

import com.naaturel.ANN.domain.abstraction.Model;
import com.naaturel.ANN.domain.abstraction.Network;
import com.naaturel.ANN.domain.model.neuron.Neuron;
import com.naaturel.ANN.domain.abstraction.Trainer;
import com.naaturel.ANN.implementation.gradientDescent.Linear;
import com.naaturel.ANN.implementation.multiLayers.Sigmoid;
import com.naaturel.ANN.implementation.multiLayers.TanH;
import com.naaturel.ANN.implementation.training.GradientBackpropagationTraining;
import com.naaturel.ANN.infrastructure.dataset.DataSet;
import com.naaturel.ANN.infrastructure.dataset.DataSetEntry;
import com.naaturel.ANN.infrastructure.dataset.DatasetExtractor;
import com.naaturel.ANN.domain.model.neuron.*;
import com.naaturel.ANN.infrastructure.graph.GraphVisualizer;

import java.io.Console;
import java.util.*;

public class Main {

    public static void main(String[] args){

        int nbrClass = 1;

        DataSet dataset = new DatasetExtractor()
                .extract("C:/Users/Laurent/Desktop/ANN-framework/src/main/resources/assets/table_4_12.csv", nbrClass);

        int[] neuronPerLayer = new int[]{50, 50, 25, dataset.getNbrLabels()};
        int nbrInput = dataset.getNbrInputs();

        FullyConnectedNetwork network = createNetwork(neuronPerLayer, nbrInput);

        System.out.println(network.synCount());

        Trainer trainer = new GradientBackpropagationTraining();
        trainer.train(0.001F, 2000, network, dataset);

        //plotGraph(dataset, network);
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

        float min = -3F;
        float max = 3F;
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
