package com.example.yin.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Service
@Slf4j
public class CNNPredictionService {
    private static final String PYTHON_EXECUTABLE = "C:\\Users\\CC\\miniconda3\\envs\\jak_pred\\python.exe";
    private static final String CNN_SCRIPT_PATH = "D:\\Desktop\\backend\\back\\server\\model_files\\cnn_predict.py";
    private static final int PYTHON_TIMEOUT_SECONDS = 120;

    @Autowired
    private ImageGenerationService imageGenerationService;

    private final ObjectMapper objectMapper = new ObjectMapper();

    public Map<String, Object> predictWithCNN(String smiles, String imageUrl) {
        validateInput(smiles);
        ProcessBuilder pb = buildProcess(smiles, imageUrl);

        try {
            Process process = pb.start();
            String jsonOutput = captureOutput(process);
            Map<String, Object> result = parseOutput(jsonOutput);

            result.put("used_provided_image", imageUrl != null);
            return result;
        } catch (Exception e) {
            throw handleExecutionError(e);
        }
    }

    private ProcessBuilder buildProcess(String smiles, String imageUrl) {
        Path pythonPath = Paths.get(PYTHON_EXECUTABLE).toAbsolutePath();
        Path scriptPath = Paths.get(CNN_SCRIPT_PATH).toAbsolutePath();
        validatePaths(pythonPath, scriptPath);

        ProcessBuilder pb = new ProcessBuilder(
                pythonPath.toString(),
                scriptPath.toString(),
                "--smiles", smiles
        );

        if (imageUrl != null && !imageUrl.isEmpty()) {
            pb.command().add("--image_url");
            pb.command().add(imageUrl);
        }

        pb.directory(scriptPath.getParent().toFile());
        pb.redirectErrorStream(true);
        return pb;
    }

    private void validateInput(String smiles) {
        if (smiles == null || smiles.trim().isEmpty()) {
            throw new IllegalArgumentException("SMILES字符串不能为空");
        }
        if (smiles.length() > 500) {
            throw new IllegalArgumentException("SMILES字符串过长");
        }
    }

    private void validatePaths(Path pythonPath, Path scriptPath) {
        if (!pythonPath.toFile().exists()) {
            throw new RuntimeException("Python解释器不存在: " + pythonPath);
        }
        if (!scriptPath.toFile().exists()) {
            throw new RuntimeException("CNN预测脚本不存在: " + scriptPath);
        }
    }

    private String captureOutput(Process process) throws IOException, InterruptedException {
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().startsWith("{")) {
                    output.setLength(0);
                    output.append(line.trim());
                } else {
                    log.debug("CNN预测输出: {}", line);
                }
            }
        }

        if (!process.waitFor(PYTHON_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            throw new RuntimeException("CNN预测超时（超过" + PYTHON_TIMEOUT_SECONDS + "秒）");
        }

        if (process.exitValue() != 0) {
            throw new RuntimeException("CNN预测脚本执行失败，退出码: " + process.exitValue() +
                    "\n输出内容: " + output);
        }

        return output.toString();
    }

    private Map<String, Object> parseOutput(String jsonOutput) {
        try {
            Map<String, Object> rawResult = objectMapper.readValue(
                    jsonOutput,
                    new TypeReference<Map<String, Object>>() {}
            );

            if (!rawResult.containsKey("data") ||
                    !((Map<?, ?>) rawResult.get("data")).containsKey("cnn_predictions")) {
                throw new RuntimeException("无效的CNN响应结构，无法解析预测结果");
            }

            Map<String, Object> result = new HashMap<>();
            result.put("cnn_predictions", rawResult.get("data"));
            result.put("status", rawResult.getOrDefault("status", "success"));
            result.put("timestamp", System.currentTimeMillis());
            return result;
        } catch (Exception e) {
            throw new RuntimeException("CNN预测结果解析失败: " + e.getMessage(), e);
        }
    }

    private RuntimeException handleExecutionError(Exception e) {
        if (e instanceof InterruptedException) {
            Thread.currentThread().interrupt();
            return new RuntimeException("CNN预测被中断", e);
        }
        return new RuntimeException("CNN预测服务异常: " + e.getMessage(), e);
    }
}