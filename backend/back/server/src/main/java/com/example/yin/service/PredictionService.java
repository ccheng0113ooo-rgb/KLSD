package com.example.yin.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Service
@Slf4j
public class PredictionService {

    private static final String PYTHON_EXECUTABLE = "C:\\Users\\CC\\miniconda3\\envs\\jak_pred\\python.exe";
    private static final String PYTHON_SCRIPT_PATH = "D:\\Desktop\\backend\\back\\server\\model_files\\predict.py";
    private static final int PYTHON_TIMEOUT_SECONDS = 60;

    private final ObjectMapper objectMapper = new ObjectMapper();

    public Map<String, Object> predictActivity(String smiles) {
        log.info("开始预测 SMILES: {}", smiles);

        // 1. 输入验证
        validateInput(smiles);

        // 2. 路径验证
        validatePaths();

        // 3. 执行Python预测
        String jsonOutput = executePythonScript(smiles);

        // 4. 解析和处理结果
        return processPredictionResult(jsonOutput);
    }

    private void validateInput(String smiles) {
        if (smiles == null || smiles.trim().isEmpty()) {
            throw new IllegalArgumentException("SMILES字符串不能为空");
        }
        // 简单SMILES格式验证
        if (!smiles.matches("^[a-zA-Z0-9@+\\-\\[\\]()/=#$.]+$")) {
            throw new IllegalArgumentException("无效的SMILES格式");
        }
    }

    private void validatePaths() {
        Path pythonPath = Paths.get(PYTHON_EXECUTABLE).toAbsolutePath();
        Path scriptPath = Paths.get(PYTHON_SCRIPT_PATH).toAbsolutePath();

        if (!pythonPath.toFile().exists()) {
            throw new RuntimeException("Python解释器不存在: " + pythonPath);
        }
        if (!scriptPath.toFile().exists()) {
            throw new RuntimeException("Python脚本不存在: " + scriptPath);
        }
    }

    private String executePythonScript(String smiles) {
        ProcessBuilder pb = new ProcessBuilder(
                PYTHON_EXECUTABLE,
                PYTHON_SCRIPT_PATH,
                "\"" + smiles + "\""
        );
        pb.directory(Paths.get(PYTHON_SCRIPT_PATH).getParent().toFile());
        pb.redirectErrorStream(true);

        try {
            Process process = pb.start();
            StringBuilder output = new StringBuilder();

            // 使用线程读取输出，防止阻塞
            Thread outputThread = new Thread(() -> {
                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()))) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        if (line.trim().startsWith("{") && line.trim().endsWith("}")) {
                            output.setLength(0);
                            output.append(line);
                        }
                        log.debug("Python输出: {}", line);
                    }
                } catch (IOException e) {
                    log.error("读取Python输出失败", e);
                }
            });
            outputThread.start();

            // 等待进程完成
            if (!process.waitFor(PYTHON_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                process.destroyForcibly();
                throw new RuntimeException("Python脚本执行超时");
            }
            outputThread.join(5000); // 等待输出线程完成

            // 检查退出码
            if (process.exitValue() != 0) {
                throw new RuntimeException(String.format(
                        "Python脚本执行失败 (code=%d): %s",
                        process.exitValue(),
                        output.length() > 0 ? output.toString() : "无错误信息"
                ));
            }

            if (output.length() == 0) {
                throw new RuntimeException("Python脚本未返回有效JSON数据");
            }

            return output.toString();

        } catch (IOException | InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("执行Python脚本失败: " + e.getMessage(), e);
        }
    }

    private Map<String, Object> processPredictionResult(String jsonOutput) {
        try {
            // 1. 原始JSON解析
            Map<String, Object> rawResult = objectMapper.readValue(
                    jsonOutput,
                    new TypeReference<Map<String, Object>>() {}
            );

            // 2. 错误处理
            if (rawResult.containsKey("error") && rawResult.get("error") != null) {
                throw new RuntimeException(rawResult.get("error").toString());
            }

            // 3. 结果标准化
            Map<String, Object> formattedResult = new HashMap<>();

            // 3.1 处理results字段
            if (rawResult.containsKey("results")) {
                Map<String, Object> results = (Map<String, Object>) rawResult.get("results");
                Map<String, Object> formattedPredictions = new HashMap<>();

                // 确保所有靶点都有结果
                for (String target : new String[]{"jak1", "jak2", "jak3", "tyk2"}) {
                    if (results.containsKey(target)) {
                        Map<String, Object> targetData = (Map<String, Object>) results.get(target);
                        formattedPredictions.put(target.toUpperCase(), targetData);
                    } else {
                        // 填充默认值
                        Map<String, Object> defaultPrediction = new HashMap<>();
                        defaultPrediction.put("predicted_activity", 0.0);
                        defaultPrediction.put("is_active", false);
                        defaultPrediction.put("error", "Target not predicted");
                        formattedPredictions.put(target.toUpperCase(), defaultPrediction);
                    }
                }
                formattedResult.put("predictions", formattedPredictions);
            }

            // 3.2 添加元数据
            formattedResult.put("success", true);
            formattedResult.put("timestamp", System.currentTimeMillis());

            log.info("预测成功完成");
            return formattedResult;

        } catch (Exception e) {
            throw new RuntimeException("处理预测结果失败: " + e.getMessage(), e);
        }
    }
}