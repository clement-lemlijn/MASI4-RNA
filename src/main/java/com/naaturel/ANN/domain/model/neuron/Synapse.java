package com.naaturel.ANN.domain.model.neuron;

public class Synapse {

    private Input input;
    private Weight weight;

    public Synapse(Input input, Weight weight){
        this.input = input;
        this.weight = weight;
    }

    public float getInput(){
        return this.input.getValue();
    }

    public void setInput(Input input){
        this.input.setValue(input.getValue());
    }

    public float getWeight() {
        return weight.getValue();
    }

    public void setWeight(float value){
        this.weight.setValue(value);
    }
}
