package com.example.yin.service;
import org.springframework.beans.factory.annotation.Value; // 添加这行
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
import javax.annotation.PostConstruct;  // 添加这行
@Service
@Slf4j
public class TDTPredictionService {
    // 从配置文件注入（替换原来的 static final）
    @Value("${tdt.python.path}")
    private String pythonExecutable;

    @Value("${tdt.script.path}")
    private String scriptPath;

    @Value("${tdt.timeout.seconds}")
    private int timeoutSeconds;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @PostConstruct
    public void init() {
        log.info("TDT Model Config:\n  Python: {}\n  Script: {}\n  Timeout: {}s",
                pythonExecutable, scriptPath, timeoutSeconds);
    }
    public Map<String, Object> predictWithTDT(String smiles) {
        validateInput(smiles);
        ProcessBuilder pb = buildProcess(smiles);

        try {
            Process process = pb.start();
            String jsonOutput = captureOutput(process);
            return parseOutput(jsonOutput);
        } catch (Exception e) {
            throw handleExecutionError(e);
        }
    }

    private ProcessBuilder buildProcess(String smiles) {
        Path pythonPath = Paths.get(this.pythonExecutable).toAbsolutePath();  // 使用注入的配置
        Path scriptPath = Paths.get(this.scriptPath).toAbsolutePath();
        validatePaths(pythonPath, scriptPath);

        ProcessBuilder pb = new ProcessBuilder(
                pythonPath.toString(),
                scriptPath.toString(),
                "--smiles", smiles
        );

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
            throw new RuntimeException("传统机器学习预测脚本不存在: " + scriptPath);
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
                    log.debug("TDT输出: {}", line);
                }
            }
        }

        // 修改这里：使用 timeoutSeconds 替代 PYTHON_TIMEOUT_SECONDS
        if (!process.waitFor(this.timeoutSeconds, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            throw new RuntimeException("传统机器学习预测超时");
        }

        if (process.exitValue() != 0) {
            throw new RuntimeException("传统机器学习执行失败: " + output);
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
                    !((Map<?, ?>) rawResult.get("data")).containsKey("tdt_predictions")) {
                throw new RuntimeException("无效的传统机器学习响应结构");
            }

            Map<String, Object> result = new HashMap<>();
            result.put("tdt_predictions", rawResult.get("data"));
            result.put("status", rawResult.getOrDefault("status", "success"));
            result.put("timestamp", System.currentTimeMillis());
            return result;
        } catch (Exception e) {
            throw new RuntimeException("传统机器学习结果解析失败: " + e.getMessage(), e);
        }
    }

    private RuntimeException handleExecutionError(Exception e) {
        if (e instanceof InterruptedException) {
            Thread.currentThread().interrupt();
            return new RuntimeException("传统机器学习预测被中断", e);
        }
        return new RuntimeException("传统机器学习预测失败: " + e.getMessage(), e);
    }
}