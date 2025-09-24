package com.example.yin.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.filter.CorsFilter;
import java.util.Arrays;
@Configuration
public class CorsConfig {

    private CorsConfiguration buildConfig() {
        CorsConfiguration config = new CorsConfiguration();

        // 1. 允许的源（开发+生产环境）
        config.setAllowedOrigins(Arrays.asList(
                "http://localhost:8080",
                "http://localhost:8889",
                "http://ai.njucm.edu.cn"
        ));
        // 2. 允许的HTTP方法（显式列出）
        config.addAllowedMethod("OPTIONS"); // 必须显式声明OPTIONS
        config.addAllowedMethod("POST");
        config.addAllowedMethod("GET");
        config.addAllowedMethod("PUT");
        config.addAllowedMethod("DELETE");

        // 3. 允许的请求头
        config.addAllowedHeader("Authorization");
        config.addAllowedHeader("Content-Type");
        config.addAllowedHeader("X-Requested-With");
        config.addAllowedHeader("Accept");

        // 4. 其他配置
        config.setAllowCredentials(true); // 允许携带cookie
        config.setMaxAge(3600L); // 预检请求缓存时间（秒）

        return config;
    }

    @Bean
    public CorsFilter corsFilter() {
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();

        // 对所有接口路径应用CORS配置
        source.registerCorsConfiguration("/**", buildConfig());

        return new CorsFilter(source);
    }
}