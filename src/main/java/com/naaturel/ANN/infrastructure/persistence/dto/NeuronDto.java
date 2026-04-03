package com.naaturel.ANN.infrastructure.persistence.dto;

import java.util.Map;

public record NeuronDto (
        int id,
        Map<Integer, Float> weights
) { }
