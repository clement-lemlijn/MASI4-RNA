package com.naaturel.ANN.domain.model.neuron;
import com.naaturel.ANN.domain.abstraction.ActivationFunction;
import com.naaturel.ANN.domain.abstraction.Model;

import java.util.List;
import java.util.function.Consumer;

public class Neuron implements Model {

    private final int id;
    private float output;
    private final float[] weights;
    private final float[] inputs;
    private final ActivationFunction activationFunction;

    public Neuron(int id, Synapse[] synapses, Bias bias, ActivationFunction func){
        this.id = id;
        this.activationFunction = func;

        output = 0;
        weights = new float[synapses.length+1]; //takes the bias into account
        inputs = new float[synapses.length+1]; //takes the bias into account

        weights[0] = bias.getWeight();
        inputs[0] = bias.getInput();
        for (int i = 0; i < synapses.length; i++){
            weights[i+1] = synapses[i].getWeight();
            inputs[i+1] = synapses[i].getInput();
        }
    }

    public void setWeight(int index, float value) {
        this.weights[index] = value;
    }

    public float getWeight(int index) {
        return this.weights[index];
    }

    public float getInput(int index) {
        return this.inputs[index];
    }

    public ActivationFunction getActivationFunction(){
        return this.activationFunction;
    }

    public float calculateWeightedSum() {
        int count = weights.length;
        float weightedSum = 0F;
        for (int i = 0; i < count; i++){
            weightedSum += weights[i] * inputs[i];
        }
        return weightedSum;
    }

    public int getId(){
        return this.id;
    }

    public float getOutput() {
        return this.output;
    }

    @Override
    public int synCount() {
        return this.weights.length;
    }

    @Override
    public int neuronCount() {
        return 1;
    }

    @Override
    public int indexInLayerOf(Neuron n) {
        return 0;
    }

    @Override
    public float[] predict(float[] inputs) {
        this.setInputs(inputs);
        output = activationFunction.accept(this);
        return new float[] {output};
    }

    @Override
    public void forEachNeuron(Consumer<Neuron> consumer) {
        consumer.accept(this);
    }

    @Override
    public void forEachOutputNeurons(Consumer<Neuron> consumer) {
        consumer.accept(this);
    }

    @Override
    public void forEachNeuronConnectedTo(Neuron n, Consumer<Neuron> consumer) {
        throw new UnsupportedOperationException("Neurons have no connection with themselves");
    }

    private void setInputs(float[] values){
        System.arraycopy(values, 0, inputs, 1, values.length);
    }

}
