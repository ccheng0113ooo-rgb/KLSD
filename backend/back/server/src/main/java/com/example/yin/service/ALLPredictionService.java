package com.example.yin.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Service
@Slf4j
public class ALLPredictionService {

    @Value("${overall.python.executable}")
    private String pythonExecutable;

    @Value("${overall.python.script.path}")
    private String pythonScriptPath;

    @Value("${overall.timeout.seconds}")
    private int timeoutSeconds;

    private final ObjectMapper objectMapper = new ObjectMapper();

    public Map<String, Object> predictActivity(String smiles) {
        log.info("Starting overall prediction for SMILES: {}", smiles);
        validateInput(smiles);
        validatePaths();
        String jsonOutput = executePythonScript(smiles);
        return processPredictionResult(jsonOutput);
    }

    private void validateInput(String smiles) {
        if (smiles == null || smiles.trim().isEmpty()) {
            throw new IllegalArgumentException("SMILES string cannot be empty");
        }
        if (!smiles.matches("^[a-zA-Z0-9@+\\-\\[\\]()/=#$.]+$")) {
            throw new IllegalArgumentException("Invalid SMILES format");
        }
    }

    private void validatePaths() {
        Path pythonPath = Paths.get(pythonExecutable).toAbsolutePath();
        Path scriptPath = Paths.get(pythonScriptPath).toAbsolutePath();

        if (!pythonPath.toFile().exists()) {
            throw new RuntimeException("Python interpreter not found: " + pythonPath);
        }
        if (!scriptPath.toFile().exists()) {
            throw new RuntimeException("Python script not found: " + scriptPath);
        }
    }

    private String executePythonScript(String smiles) {
        ProcessBuilder pb = new ProcessBuilder(
                pythonExecutable,
                "-u",  // 无缓冲输出
                pythonScriptPath,
                smiles
        );
        pb.directory(Paths.get(pythonScriptPath).getParent().toFile());
        pb.redirectErrorStream(true);

        try {
            Process process = pb.start();
            StringBuilder jsonOutput = new StringBuilder();

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {

                String line;
                while ((line = reader.readLine()) != null) {
                    // 只捕获JSON格式的输出
                    if (line.trim().startsWith("{") && line.trim().endsWith("}")) {
                        jsonOutput.append(line.trim());
                        break; // 找到JSON后立即退出
                    }
                    log.debug("Python output: {}", line);
                }
            }

            if (!process.waitFor(timeoutSeconds, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                throw new RuntimeException("Python script execution timeout");
            }

            if (jsonOutput.length() == 0) {
                throw new RuntimeException("Python script did not return valid JSON data");
            }

            // 验证JSON格式
            try {
                objectMapper.readTree(jsonOutput.toString());
                return jsonOutput.toString();
            } catch (Exception e) {
                throw new RuntimeException("Invalid JSON output: " + jsonOutput, e);
            }

        } catch (IOException | InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Failed to execute Python script: " + e.getMessage(), e);
        }
    }

    private Map<String, Object> processPredictionResult(String jsonOutput) {
        try {
            Map<String, Object> rawResult = objectMapper.readValue(
                    jsonOutput,
                    new TypeReference<Map<String, Object>>() {}
            );

            if (!Boolean.TRUE.equals(rawResult.get("success"))) {
                throw new RuntimeException(rawResult.get("error").toString());
            }

            Map<String, Object> results = (Map<String, Object>) rawResult.get("results");
            Map<String, Object> overall = (Map<String, Object>) results.get("overall");

            Map<String, Object> formattedPrediction = new HashMap<>();
            Object activityValue = overall.get("predicted_activity");
            Object isActiveValue = overall.get("is_active");

            if (activityValue instanceof List) {
                List<?> activityList = (List<?>) activityValue;
                if (!activityList.isEmpty()) {
                    formattedPrediction.put("predicted_activity", activityList.get(0));
                    formattedPrediction.put("is_active", ((List<?>) isActiveValue).get(0));
                }
            } else {
                formattedPrediction.put("predicted_activity", activityValue);
                formattedPrediction.put("is_active", isActiveValue);
            }

            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("prediction", formattedPrediction);
            response.put("timestamp", System.currentTimeMillis());

            if (rawResult.containsKey("errors")) {
                response.put("warnings", rawResult.get("errors"));
            }

            return response;
        } catch (Exception e) {
            throw new RuntimeException("Failed to process prediction result: " + e.getMessage(), e);
        }
    }
}