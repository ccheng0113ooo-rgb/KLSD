package com.example.yin.controller;

import com.example.yin.service.ImageGenerationService;
import com.example.yin.service.CNNPredictionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class CNNPredictionController {

    private final CNNPredictionService cnnPredictionService;
    private final ImageGenerationService imageGenerationService;

    @Autowired
    public CNNPredictionController(CNNPredictionService cnnPredictionService,
                                   ImageGenerationService imageGenerationService) {
        this.cnnPredictionService = cnnPredictionService;
        this.imageGenerationService = imageGenerationService;
    }

    @PostMapping("/cnn-predict")
    public ResponseEntity<Map<String, Object>> cnnPredict(@RequestBody Map<String, String> request) {
        String smiles = request.get("smiles");
        Map<String, Object> response = new HashMap<>();
        response.put("timestamp", System.currentTimeMillis());

        try {
            String imageUrl = imageGenerationService.generateImage(smiles);
            Map<String, Object> result = cnnPredictionService.predictWithCNN(smiles, imageUrl);

            response.put("status", HttpStatus.OK.value());
            response.put("data", result);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            response.put("status", HttpStatus.INTERNAL_SERVER_ERROR.value());
            response.put("error", "CNN预测失败: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }
    }
}