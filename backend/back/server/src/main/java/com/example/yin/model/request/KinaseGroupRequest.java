package com.example.yin.model.request;

import lombok.Data;

@Data
public class KinaseGroupRequest {

    private Integer groupId;

    private String groupName;

    private String familyName;

    private String subfamilyName;

    private Integer number;

    private Integer active;

    private Integer inactive;

    private String comment;

}