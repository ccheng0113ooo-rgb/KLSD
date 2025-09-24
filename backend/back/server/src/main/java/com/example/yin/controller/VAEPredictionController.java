package com.example.yin.controller;

import com.example.yin.service.ImageGenerationService;
import com.example.yin.service.VaePredictionService;
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
public class VAEPredictionController {

    private final VaePredictionService vaePredictionService;
    private final ImageGenerationService imageGenerationService;

    @Autowired
    public VAEPredictionController(VaePredictionService vaePredictionService,
                                   ImageGenerationService imageGenerationService) {
        this.vaePredictionService = vaePredictionService;
        this.imageGenerationService = imageGenerationService;
    }

    @PostMapping("/vae-predict")
    public ResponseEntity<Map<String, Object>> vaePredict(@RequestBody Map<String, String> request) {
        String smiles = request.get("smiles");
        Map<String, Object> response = new HashMap<>();
        response.put("timestamp", System.currentTimeMillis());

        try {
            // 1. 获取或生成分子图像（保留代码，但不再用于预测）
            String imageUrl = imageGenerationService.generateImage(smiles);

            // 2. 使用VAE模型进行预测（只传递smiles参数）
            Map<String, Object> result = vaePredictionService.predictWithVAE(smiles);

            // 3. 构建响应
            response.put("status", HttpStatus.OK.value());
            response.put("data", result);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            response.put("status", HttpStatus.INTERNAL_SERVER_ERROR.value());
            response.put("error", "VAE预测失败: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }
    }
}