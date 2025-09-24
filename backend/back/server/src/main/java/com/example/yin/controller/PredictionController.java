
//package com.example.yin.controller;
//
//import com.example.yin.service.PredictionService;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//import org.springframework.beans.factory.annotation.Autowired;
//import org.springframework.http.HttpStatus;
//import org.springframework.http.MediaType;
//import org.springframework.http.ResponseEntity;
//import org.springframework.web.bind.annotation.*;
//
//import java.util.HashMap;
//import java.util.Map;
//
///**
// * API端点：化合物活性预测
// * 基础路径：/api/predict
// */
//@RestController
//@RequestMapping("/api")  // 类级别路由前缀
//@CrossOrigin(origins = "http://localhost:8080")  // 明确允许前端跨域访问
//public class PredictionController {
//
//    private static final Logger logger = LoggerFactory.getLogger(PredictionController.class);
//
//    @Autowired
//    private PredictionService predictionService;
//
//    /**
//     * POST /api/predict
//     * 接收SMILES字符串并返回预测结果
//     *
//     * @param request 必须包含"smiles"字段
//     * @return 标准化JSON响应：
//     *         - 成功: {status:200, data:{...}, timestamp:...}
//     *         - 失败: {status:4xx/5xx, error:"...", timestamp:...}
//     */
//    @PostMapping(
//            path = "/predict",
//            consumes = MediaType.APPLICATION_JSON_VALUE,
//            produces = MediaType.APPLICATION_JSON_VALUE
//    )
//    public ResponseEntity<Map<String, Object>> predict(
//            @RequestBody Map<String, String> request
//    ) {
//        // 1. 请求日志记录
//        logRequestDetails(request);
//
//        // 2. 构建基础响应
//        Map<String, Object> response = new HashMap<>();
//        response.put("timestamp", System.currentTimeMillis());
//
//        // 3. 输入验证
//        if (!validateRequest(request, response)) {
//            return buildErrorResponse(response, HttpStatus.BAD_REQUEST);
//        }
//
//        // 4. 执行预测
//        try {
//            String smiles = request.get("smiles").trim();
//            logger.info("Starting prediction for: {}", smiles);
//
//            Map<String, Object> result = predictionService.predictActivity(smiles);
//            logger.info("Prediction succeeded for: {}", smiles);
//
//            // 5. 构建成功响应
//            response.put("status", HttpStatus.OK.value());
//            response.put("data", result);
//
//            return ResponseEntity.ok()
//                    .contentType(MediaType.APPLICATION_JSON)
//                    .body(response);
//
//        } catch (Exception e) {
//            // 6. 错误处理
//            return handlePredictionError(e, response);
//        }
//    }
//
//    // ========== 私有方法 ========== //
//
//    /**
//     * 记录请求详情
//     */
//    private void logRequestDetails(Map<String, String> request) {
//        logger.info("Received prediction request");
//        if (logger.isDebugEnabled()) {
//            logger.debug("Request details: {}", request);
//            logger.debug("SMILES: {}", request.getOrDefault("smiles", "null"));
//        }
//    }
//
//    /**
//     * 验证请求参数
//     */
//    private boolean validateRequest(
//            Map<String, String> request,
//            Map<String, Object> response
//    ) {
//        if (!request.containsKey("smiles")) {
//            response.put("error", "Missing required field: 'smiles'");
//            return false;
//        }
//
//        String smiles = request.get("smiles").trim();
//        if (smiles.isEmpty()) {
//            response.put("error", "SMILES string cannot be empty");
//            return false;
//        }
//
//        return true;
//    }
//
//    /**
//     * 构建错误响应
//     */
//    private ResponseEntity<Map<String, Object>> buildErrorResponse(
//            Map<String, Object> response,
//            HttpStatus status
//    ) {
//        response.put("status", status.value());
//        logger.warn("Request validation failed: {}", response.get("error"));
//
//        return ResponseEntity.status(status)
//                .contentType(MediaType.APPLICATION_JSON)
//                .body(response);
//    }
//
//    /**
//     * 处理预测异常
//     */
//    private ResponseEntity<Map<String, Object>> handlePredictionError(
//            Exception e,
//            Map<String, Object> response
//    ) {
//        String errorMsg = "Prediction failed: " + e.getMessage();
//        logger.error(errorMsg, e);
//
//        response.put("status", HttpStatus.INTERNAL_SERVER_ERROR.value());
//        response.put("error", errorMsg);
//
//        // 开发环境添加堆栈跟踪
//        if (logger.isDebugEnabled()) {
//            response.put("debug", e.getStackTrace());
//        }
//
//        return ResponseEntity.internalServerError()
//                .contentType(MediaType.APPLICATION_JSON)
//                .body(response);
//    }
//}



package com.example.yin.controller;

import com.example.yin.service.PredictionService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "http://localhost:8080")
public class PredictionController {

    private static final Logger logger = LoggerFactory.getLogger(PredictionController.class);

    @Autowired
    private PredictionService predictionService;

    @PostMapping(
            path = "/predict",
            consumes = MediaType.APPLICATION_JSON_VALUE,
            produces = MediaType.APPLICATION_JSON_VALUE
    )
    public ResponseEntity<Map<String, Object>> predict(
            @RequestBody Map<String, String> request
    ) {
        // 1. 请求日志记录
        logRequestDetails(request);

        // 2. 构建基础响应
        Map<String, Object> response = new HashMap<>();
        response.put("timestamp", System.currentTimeMillis());

        // 3. 输入验证
        if (!validateRequest(request, response)) {
            return buildErrorResponse(response, HttpStatus.BAD_REQUEST);
        }

        // 4. 执行预测
        try {
            String smiles = request.get("smiles").trim();
            logger.info("Starting prediction for: {}", smiles);

            Map<String, Object> result = predictionService.predictActivity(smiles);
            logger.info("Prediction succeeded for: {}", smiles);

            // 5. 构建成功响应
            response.put("status", HttpStatus.OK.value());
            response.put("data", result);

            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(response);

        } catch (Exception e) {
            // 6. 错误处理
            return handlePredictionError(e, response);
        }
    }

    private void logRequestDetails(Map<String, String> request) {
        logger.info("Received prediction request");
        if (logger.isDebugEnabled()) {
            logger.debug("Request details: {}", request);
            logger.debug("SMILES: {}", request.getOrDefault("smiles", "null"));
        }
    }

    private boolean validateRequest(
            Map<String, String> request,
            Map<String, Object> response
    ) {
        if (!request.containsKey("smiles")) {
            response.put("error", "Missing required field: 'smiles'");
            return false;
        }

        String smiles = request.get("smiles").trim();
        if (smiles.isEmpty()) {
            response.put("error", "SMILES string cannot be empty");
            return false;
        }

        return true;
    }

    private ResponseEntity<Map<String, Object>> buildErrorResponse(
            Map<String, Object> response,
            HttpStatus status
    ) {
        response.put("status", status.value());
        logger.warn("Request validation failed: {}", response.get("error"));

        return ResponseEntity.status(status)
                .contentType(MediaType.APPLICATION_JSON)
                .body(response);
    }

    private ResponseEntity<Map<String, Object>> handlePredictionError(
            Exception e,
            Map<String, Object> response
    ) {
        String errorMsg = "Prediction failed: " + e.getMessage();
        logger.error(errorMsg, e);

        response.put("status", HttpStatus.INTERNAL_SERVER_ERROR.value());
        response.put("error", errorMsg);

        if (logger.isDebugEnabled()) {
            response.put("debug", e.getStackTrace());
        }

        return ResponseEntity.internalServerError()
                .contentType(MediaType.APPLICATION_JSON)
                .body(response);
    }
}