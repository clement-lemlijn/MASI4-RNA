package com.naaturel.ANN.domain.model.neuron;

public class Input {

    private float value;

    public Input(float value){
        this.value = value;
    }

    public void setValue(float value){
        this.value = value;
    }

    public float getValue(){
        return this.value;
    }


}
