package com.example.yin.model.request;
import lombok.Data;

@Data
public class TargetRequest {
    private String moleculeChemblId;

    private String targetname1;

    private String targetname2;

    private Double diff;

    private Double pact1;

    private Double pact2;

    private String documentChemblId1;

    private String documentChemblId2;
}
