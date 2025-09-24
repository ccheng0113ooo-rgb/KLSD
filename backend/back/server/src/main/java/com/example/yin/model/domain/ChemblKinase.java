package com.example.yin.model.domain;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import org.apache.commons.lang3.builder.ToStringBuilder;

import java.io.Serializable;

@TableName(value = "chembl_kinase")
@Data
public class ChemblKinase implements Serializable {
    @TableId(type = IdType.AUTO)
    private Integer kinaseid;
    private String moleculechemblid;
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
    private String targetname;
    private String targetorganism;
    private String targettype;
    private String documentchemblid;
    private String sourcedescription;
    private byte[] structureImgBinary;

    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this);
    }
}
