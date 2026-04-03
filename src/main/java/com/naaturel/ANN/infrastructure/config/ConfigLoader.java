package com.naaturel.ANN.infrastructure.config;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;

public class ConfigLoader {


    public static ConfigDto load(String path) throws Exception {
        try {

            ObjectMapper mapper = new ObjectMapper();
            ConfigDto config = mapper.readValue(new File("config.json"), ConfigDto.class);

            return config;
        } catch (Exception e){
            throw new Exception("Unable to load config : " + e.getMessage());
        }
    }

}
