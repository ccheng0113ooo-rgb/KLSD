package com.example.yin.model.request;
import lombok.Data;

@Data
public class CompoundRequest {
    private Integer compoundId;

    private String moleculeChemblId;

    private String name;

    private String standardType;

    private String standardRelation;

    private String standardValue;

    private String standardUnits;

    private String documentChemblId;

}
