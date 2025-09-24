package com.example.yin.model.request;
import lombok.Data;

@Data
public class DrugsRequest {
    private Integer drugId;

    private String moleculeChemblId;

    private String drugName;

    private String name;

    private String standardType;

    private String standardRelation;

    private String standardValue;

    private String standardUnits;

    private String documentChemblId;

}
