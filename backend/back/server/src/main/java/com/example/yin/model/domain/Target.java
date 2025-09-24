package com.example.yin.model.domain;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import org.apache.commons.lang3.builder.ToStringBuilder;

import java.io.Serializable;

@TableName(value = "target")
@Data
public class Target implements Serializable{
    private String moleculeChemblId;
    private String targetname1;
    private String targetname2;
    private Double diff;
    private Double pact1;
    private Double pact2;
    private String documentChemblId1;
    private String documentChemblId2;

    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this);
    }
}
