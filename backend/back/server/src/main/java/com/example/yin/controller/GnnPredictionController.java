package com.example.yin.controller;

import com.example.yin.service.GnnPredictionService;
import com.example.yin.service.ImageGenerationService;
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
public class GnnPredictionController {

    private final GnnPredictionService gnnPredictionService;
    private final ImageGenerationService imageGenerationService;

    @Autowired
    public GnnPredictionController(GnnPredictionService gnnPredictionService,
                                   ImageGenerationService imageGenerationService) {
        this.gnnPredictionService = gnnPredictionService;
        this.imageGenerationService = imageGenerationService;
    }

    @PostMapping("/gnn-predict")
    public ResponseEntity<Map<String, Object>> gnnPredict(@RequestBody Map<String, String> request) {
        String smiles = request.get("smiles");
        Map<String, Object> response = new HashMap<>();
        response.put("timestamp", System.currentTimeMillis());

        try {
            // 1. 获取或生成分子图像
            String imageUrl = imageGenerationService.generateImage(smiles);

            // 2. 使用图像进行GNN预测
            Map<String, Object> result = gnnPredictionService.predictWithGNN(smiles, imageUrl);

            // 3. 构建响应
            response.put("status", HttpStatus.OK.value());
            response.put("data", result);
            return ResponseEntity.ok(response);

        } catch (Exception e) {
            response.put("status", HttpStatus.INTERNAL_SERVER_ERROR.value());
            response.put("error", "GNN预测失败: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }
    }
}