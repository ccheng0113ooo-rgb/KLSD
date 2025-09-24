package com.example.yin.model.domain;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import org.apache.commons.lang3.builder.ToStringBuilder;

import java.io.Serializable;

@TableName(value = "kinase_group")
@Data
public class KinaseGroup implements Serializable{
    @TableId(type = IdType.AUTO)
    private Integer groupid;
    private String groupname;
    private String familyname;
    private String subfamilyname;
    private Integer number;
    private Integer active;
    private Integer inactive;
    private String comment;

    @Override
    public String toString() {
        return ToStringBuilder.reflectionToString(this);
    }
}
