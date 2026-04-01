package com.naaturel.ANN.domain.model.neuron;
import com.naaturel.ANN.domain.abstraction.ActivationFunction;
import com.naaturel.ANN.domain.abstraction.Model;

import java.util.List;
import java.util.function.Consumer;

public class Neuron implements Model {

    protected Synapse[] synapses;
    protected Bias bias;
    protected ActivationFunction activationFunction;
    protected Float output;
    protected Float weightedSum;

    public Neuron(Synapse[] synapses, Bias bias, ActivationFunction func){
        this.synapses = synapses;
        this.bias = bias;
        this.activationFunction = func;
        this.output = null;
        this.weightedSum = null;
    }

    public void updateBias(Weight weight) {
        this.bias.setWeight(weight.getValue());
    }

    public void updateWeight(int index, Weight weight) {
        this.synapses[index].setWeight(weight.getValue());
    }

    protected void setInputs(List<Input> inputs){
        for(int i = 0;  i < inputs.size() && i < synapses.length; i++){
            Synapse syn = this.synapses[i];
            syn.setInput(inputs.get(i));
        }
    }

    public ActivationFunction getActivationFunction(){
        return this.activationFunction;
    }

    public float getOutput(){
        return this.output;
    }

    public float getWeight(int index){
        return this.synapses[index].getWeight();
    }

    public float getWeightedSum(){
        return this.weightedSum;
    }

    public float calculateWeightedSum() {
        this.weightedSum = 0F;
        this.weightedSum += this.bias.getWeight() * this.bias.getInput();
        for(Synapse syn : this.synapses){
            this.weightedSum += syn.getWeight() * syn.getInput();
        }
        return this.weightedSum;
    }

    @Override
    public int synCount() {
        return this.synapses.length+1; //take the bias into account
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
    public List<Float> predict(List<Input> inputs) {
        this.setInputs(inputs);
        this.output = activationFunction.accept(this);
        return List.of(output);
    }

    @Override
    public void forEachNeuron(Consumer<Neuron> consumer) {
        consumer.accept(this);
    }

    @Override
    public void forEachSynapse(Consumer<Synapse> consumer) {
        consumer.accept(this.bias);
        for (Synapse syn : this.synapses){
            consumer.accept(syn);
        }
    }

    @Override
    public void forEachOutputNeurons(Consumer<Neuron> consumer) {
        consumer.accept(this);
    }

    @Override
    public void forEachNeuronConnectedTo(Neuron n, Consumer<Neuron> consumer) {
        throw new UnsupportedOperationException("Neurons have no connection with themselves");
    }
}
