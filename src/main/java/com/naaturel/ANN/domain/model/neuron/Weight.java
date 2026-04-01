package com.naaturel.ANN.domain.model.neuron;

import java.util.Random;

public class Weight {

    private float value;

    public Weight(){
        this(new Random().nextFloat() * 2 - 1);
    }

    public Weight(float value){
        this.value = value;
    }

    public void setValue(float value){
        this.value = value;
    }

    public float getValue(){
        return this.value;
    }

}
