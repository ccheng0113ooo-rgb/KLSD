package com.example.yin.controller;

import com.example.yin.service.TDTPredictionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class TDTPredictionController {

    private final TDTPredictionService tdtPredictionService;

    @Autowired
    public TDTPredictionController(TDTPredictionService tdtPredictionService) {
        this.tdtPredictionService = tdtPredictionService;
    }

    @PostMapping("/tdt-predict")
    public ResponseEntity<Map<String, Object>> predict(@RequestBody Map<String, String> request) {
        String smiles = request.get("smiles");
        return ResponseEntity.ok(tdtPredictionService.predictWithTDT(smiles));
    }
}