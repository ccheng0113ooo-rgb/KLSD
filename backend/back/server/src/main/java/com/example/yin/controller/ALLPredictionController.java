package com.example.yin.controller;

import com.example.yin.service.ALLPredictionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/all")
@CrossOrigin(origins = "http://localhost:8080")
public class ALLPredictionController {

    private final ALLPredictionService predictionService;

    @Autowired
    public ALLPredictionController(ALLPredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @PostMapping(
            path = "/predict",
            consumes = MediaType.APPLICATION_JSON_VALUE,
            produces = MediaType.APPLICATION_JSON_VALUE
    )
    public ResponseEntity<Map<String, Object>> predict(
            @RequestBody Map<String, String> request
    ) {
        Map<String, Object> response = new HashMap<>();
        response.put("timestamp", System.currentTimeMillis());

        // 输入验证
        if (!request.containsKey("smiles")) {
            response.put("error", "Missing required field: 'smiles'");
            return ResponseEntity.badRequest().body(response);
        }

        String smiles = request.get("smiles").trim();
        if (smiles.isEmpty()) {
            response.put("error", "SMILES string cannot be empty");
            return ResponseEntity.badRequest().body(response);
        }

        try {
            Map<String, Object> result = predictionService.predictActivity(smiles);
            response.put("status", HttpStatus.OK.value());
            response.put("data", result);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            response.put("status", HttpStatus.INTERNAL_SERVER_ERROR.value());
            response.put("error", "Prediction failed: " + e.getMessage());
            return ResponseEntity.internalServerError().body(response);
        }
    }
}