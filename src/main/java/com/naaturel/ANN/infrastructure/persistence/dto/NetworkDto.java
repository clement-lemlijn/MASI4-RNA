package com.naaturel.ANN.infrastructure.persistence.dto;

import java.util.List;

public record NetworkDto(
        List<LayerDto> layers
) { }
