package com.example.yin.model.request;

import lombok.Data;

@Data
public class ChemblKinaseRequest {
    private Integer kinaseid;

    private String moleculechemblId;

    private String compoundkey;

    private String smiles;

    private String standardtype;

    private String standardrelation;

    private String standardvalue;

    private String standardunits;

    private String pchemblvalue;

    private String assaychemblid;

    private String assaydescription;

    private String baolabel;

    private String assayorganism;

    private String targetchemblid;

    private String TargetName;

    private String targetorganism;

    private String targettype;

    private String documentchemblid;

    private String sourcedescription;

    private byte[] structureImgBinary;

}