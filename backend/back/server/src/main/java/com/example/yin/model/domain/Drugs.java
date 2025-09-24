package com.example.yin.model.domain;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import org.apache.commons.lang3.builder.ToStringBuilder;

import java.io.Serializable;

@TableName(value = "drugs")
@Data
public class Drugs implements Serializable{
    @TableId(type = IdType.AUTO)
    private Integer drugId;
    private String moleculeChemblId;
    private String drugName;
    private String name;
    private String standardType;
    private String standardRelation;
    private String standardValue;
    private String standardUnits;
    private String documentChemblId;

    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this);
    }
}
