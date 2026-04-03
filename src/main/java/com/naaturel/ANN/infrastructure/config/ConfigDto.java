package com.naaturel.ANN.infrastructure.config;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

public class ConfigDto {

    @JsonProperty("model")
    private Map<String, Object> modelConfig;

    @JsonProperty("training")
    private Map<String, Object> trainingConfig;

    @JsonProperty("dataset")
    private Map<String, Object> datasetConfig;

    public <T> T getModelProperty(String key, Class<T> type) {
        Object value = find(key, this.modelConfig);
        if (value instanceof List<?> list && type.isArray()) {
            int[] arr = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                arr[i] = ((Number) list.get(i)).intValue();
            }
            return type.cast(arr);
        }
        if (!type.isInstance(value)) {
            throw new RuntimeException("Property '" + key + "' is not of type " + type.getSimpleName());
        }
        return type.cast(value);
    }

    public <T> T getTrainingProperty(String key, Class<T> type) {
        Object value = find(key, this.trainingConfig);
        if (!type.isInstance(value)) {
            throw new RuntimeException("Property '" + key + "' is not of type " + type.getSimpleName());
        }
        return type.cast(value);
    }

    public <T> T getDatasetProperty(String key, Class<T> type) {
        Object value = find(key, this.datasetConfig);
        if (!type.isInstance(value)) {
            throw new RuntimeException("Property '" + key + "' is not of type " + type.getSimpleName());
        }
        return type.cast(value);
    }

    private Object find(String key, Map<String, Object> config){
        if(!config.containsKey(key)) throw new RuntimeException("Unable to find property for key '" + key + "'");
        return config.get(key);
    }
}
