package com.example.yin.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Service
@Slf4j
public class VaePredictionService {

    // 匹配配置文件中的 vae.python.path
    @Value("${vae.python.path}")
    private String pythonExecutable;

    // 匹配配置文件中的 vae.script.path
    @Value("${vae.script.path}")
    private String vaeScriptPath;

    // 匹配配置文件中的 vae.timeout.seconds
    @Value("${vae.timeout.seconds}")
    private int pythonTimeoutSeconds;

    private final ObjectMapper objectMapper = new ObjectMapper();

    public Map<String, Object> predictWithVAE(String smiles) {
        validateInput(smiles);

        log.info("Starting VAE prediction for SMILES: {}", smiles);
        ProcessBuilder pb = buildProcess(smiles);

        try {
            Process process = pb.start();
            String jsonOutput = captureJsonOutput(process);
            return parseOutput(jsonOutput);

        } catch (Exception e) {
            log.error("VAE prediction failed", e);
            throw new RuntimeException("VAE预测失败: " + e.getMessage(), e);
        }
    }

    private ProcessBuilder buildProcess(String smiles) {
        Path pythonPath = Paths.get(pythonExecutable).toAbsolutePath();
        Path scriptPath = Paths.get(vaeScriptPath).toAbsolutePath();

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
        StringBuilder fullOutput = new StringBuilder();
        StringBuilder jsonContent = new StringBuilder();

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()))) {

            String line;
            boolean inJson = false;
            int braceCount = 0;

            while ((line = reader.readLine()) != null) {
                fullOutput.append(line).append("\n");
                line = line.trim();

                // 检测JSON开始（第一个{）
                if (!inJson && line.startsWith("{")) {
                    inJson = true;
                    jsonContent.append(line);
                    braceCount += countBraces(line);
                    continue;
                }

                // 收集JSON内容
                if (inJson) {
                    jsonContent.append(line);
                    braceCount += countBraces(line);

                    // 检查是否找到匹配的}
                    if (braceCount == 0) {
                        break;
                    }
                }
            }
        }

        // 验证JSON完整性
        String jsonOutput = jsonContent.toString().trim();
        if (jsonOutput.isEmpty()) {
            throw new RuntimeException("未找到有效的JSON输出. 完整输出:\n" + fullOutput);
        }

        if (!process.waitFor(pythonTimeoutSeconds, TimeUnit.SECONDS)) {
            process.destroyForcibly();
            throw new RuntimeException("VAE预测超时. 完整输出:\n" + fullOutput);
        }

        if (process.exitValue() != 0) {
            throw new RuntimeException("VAE执行失败. 完整输出:\n" + fullOutput);
        }

        return jsonOutput;
    }

    // 辅助方法：计算字符串中{和}的数量差
    private int countBraces(String line) {
        int count = 0;
        for (char c : line.toCharArray()) {
            if (c == '{') count++;
            if (c == '}') count--;
        }
        return count;
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
            response.put("vae_predictions", result.get("data"));
            response.put("status", result.getOrDefault("status", "success"));
            response.put("timestamp", System.currentTimeMillis());

            return response;

        } catch (Exception e) {
            throw new RuntimeException("解析VAE输出失败: " + e.getMessage() +
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