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
public class GnnPredictionService {
    private static final String PYTHON_EXECUTABLE = "C:\\Users\\CC\\miniconda3\\envs\\jak_pred\\python.exe";
    private static final String GNN_SCRIPT_PATH = "D:\\Desktop\\backend\\back\\server\\model_files\\gnn_predict.py";
    private static final int PYTHON_TIMEOUT_SECONDS = 120;

    @Autowired
    private ImageGenerationService imageGenerationService;

    private final ObjectMapper objectMapper = new ObjectMapper();

    public Map<String, Object> predictWithGNN(String smiles, String imageUrl) {
        validateInput(smiles);

        log.info("Starting GNN prediction for SMILES: {}", smiles);
        ProcessBuilder pb = buildProcess(smiles, imageUrl);

        try {
            Process process = pb.start();
            String jsonOutput = captureJsonOutput(process);
            return parseOutput(jsonOutput);

        } catch (Exception e) {
            log.error("GNN prediction failed", e);
            throw new RuntimeException("GNN预测失败: " + e.getMessage(), e);
        }
    }

    private ProcessBuilder buildProcess(String smiles, String imageUrl) {
        Path pythonPath = Paths.get(PYTHON_EXECUTABLE).toAbsolutePath();
        Path scriptPath = Paths.get(GNN_SCRIPT_PATH).toAbsolutePath();

        log.info("Using Python: {}", pythonPath);
        log.info("Using script: {}", scriptPath);

        ProcessBuilder pb = new ProcessBuilder(
                pythonPath.toString(),
                scriptPath.toString(),
                "--smiles", smiles
        );

        // 设置环境变量
        Map<String, String> env = pb.environment();
        env.put("PATH", "C:\\Users\\CC\\miniconda3\\envs\\jak_pred;" +
                "C:\\Users\\CC\\miniconda3\\envs\\jak_pred\\Scripts;" +
                env.get("PATH"));

        pb.directory(scriptPath.getParent().toFile());
        pb.redirectErrorStream(true);  // 合并错误流

        return pb;
    }

    private String captureJsonOutput(Process process) throws IOException, InterruptedException {
        StringBuilder jsonOutput = new StringBuilder();
        StringBuilder fullOutput = new StringBuilder();

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()))) {

            String line;
            while ((line = reader.readLine()) != null) {
                fullOutput.append(line).append("\n");

                // 只捕获JSON行（以{开头）
                if (line.trim().startsWith("{")) {
                    jsonOutput.append(line.trim());
                }
            }
        }

        if (!process.waitFor(PYTHON_TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            throw new RuntimeException("GNN预测超时. 完整输出:\n" + fullOutput);
        }

        if (process.exitValue() != 0) {
            throw new RuntimeException("GNN执行失败. 完整输出:\n" + fullOutput);
        }

        if (jsonOutput.length() == 0) {
            throw new RuntimeException("未收到有效的JSON输出. 完整输出:\n" + fullOutput);
        }

        return jsonOutput.toString();
    }

    private Map<String, Object> parseOutput(String jsonOutput) {
        try {
            log.debug("Parsing JSON output: {}", jsonOutput);

            Map<String, Object> result = objectMapper.readValue(
                    jsonOutput,
                    new TypeReference<Map<String, Object>>() {}
            );

            if (result.containsKey("error")) {
                throw new RuntimeException("Python错误: " + result.get("error"));
            }

            if (!result.containsKey("data")) {
                throw new RuntimeException("响应缺少'data'字段");
            }

            Map<String, Object> response = new HashMap<>();
            response.put("gnn_predictions", result.get("data"));
            response.put("status", result.getOrDefault("status", "success"));
            response.put("timestamp", System.currentTimeMillis());

            return response;

        } catch (Exception e) {
            throw new RuntimeException("解析GNN输出失败: " + e.getMessage() +
                    "\n原始输出: " + jsonOutput, e);
        }
    }

    private void validateInput(String smiles) {
        if (smiles == null || smiles.trim().isEmpty()) {
            throw new IllegalArgumentException("SMILES字符串不能为空");
        }
        if (smiles.length() > 500) {
            throw new IllegalArgumentException("SMILES字符串过长（最大500字符）");
        }
    }
}