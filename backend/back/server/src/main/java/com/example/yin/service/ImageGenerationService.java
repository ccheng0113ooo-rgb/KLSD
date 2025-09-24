package com.example.yin.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class ImageGenerationService {
    private static final Logger logger = LoggerFactory.getLogger(ImageGenerationService.class);
    private final ConcurrentHashMap<String, String> imageCache = new ConcurrentHashMap<>();

    @Value("${python.command:python}")
    private String pythonCommand;

    @Value("${rdkit.script.path:src/main/resources/scripts/generate_molecule_image.py}")
    private String scriptPath;

    @Value("${image.temp.dir:temp/images}")
    private String tempImageDir;

    /**
     * 获取缓存的分子图像（如果存在）
     */
    public String getCachedImage(String smiles) {
        return imageCache.get(smiles);
    }

    /**
     * 生成或获取缓存的分子图像
     */
    public String generateImage(String smiles) {
        // 参数验证
        if (!StringUtils.hasText(smiles)) {
            throw new IllegalArgumentException("SMILES string cannot be empty");
        }

        // 检查缓存
        String cachedImage = imageCache.get(smiles);
        if (cachedImage != null) {
            logger.debug("Returning cached image for SMILES: {}", smiles);
            return cachedImage;
        }

        // 创建临时目录
        File tempDir = new File(tempImageDir);
        if (!tempDir.exists() && !tempDir.mkdirs()) {
            throw new RuntimeException("Failed to create temp directory");
        }

        // 生成唯一文件名
        String fileName = UUID.randomUUID().toString() + ".png";
        Path imagePath = Paths.get(tempImageDir, fileName);

        try {
            // 执行Python脚本生成图像
            Process process = new ProcessBuilder(
                    pythonCommand,
                    scriptPath,
                    smiles,
                    imagePath.toString()
            ).start();

            // 处理输出
            int exitCode = process.waitFor();
            String stdOutput = readStream(process.getInputStream());
            String errorOutput = readStream(process.getErrorStream());

            if (exitCode != 0) {
                logger.error("Image generation failed. Exit code: {}, Error: {}", exitCode, errorOutput);
                throw new RuntimeException("Molecule image generation failed");
            }

            if (!Files.exists(imagePath)) {
                throw new RuntimeException("Generated image file not found");
            }

            // 转换为Base64并缓存
            byte[] imageBytes = Files.readAllBytes(imagePath);
            String base64Image = "data:image/png;base64," + Base64.getEncoder().encodeToString(imageBytes);
            imageCache.put(smiles, base64Image);

            // 清理临时文件
            Files.deleteIfExists(imagePath);

            return base64Image;
        } catch (Exception e) {
            logger.error("Error generating molecule image for SMILES: {}", smiles, e);
            throw new RuntimeException("Molecule image generation error", e);
        }
    }

    /**
     * 清除图像缓存（可选）
     */
    public void clearCache() {
        imageCache.clear();
        logger.info("Image cache cleared");
    }

    /**
     * 获取当前缓存大小（监控用）
     */
    public int getCacheSize() {
        return imageCache.size();
    }

    private String readStream(InputStream inputStream) throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            StringBuilder builder = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                builder.append(line).append("\n");
            }
            return builder.toString();
        }
    }
}