package com.example.yin.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.example.yin.common.R;
import com.example.yin.model.domain.KinaseGroup;
import com.example.yin.model.request.KinaseGroupRequest;
import org.springframework.web.multipart.MultipartFile;import com.example.yin.common.R;
import com.example.yin.model.domain.KinaseGroup;
import com.example.yin.model.request.KinaseGroupRequest;

import java.util.List;

public interface KinaseGroupService extends IService<KinaseGroup> {

    R updateKinaseGroupMsg(KinaseGroupRequest updateKinaseGroupRequest);

    R allKinaseGroup();

    R likeGroupName(String groupName);

    R likeSubfamilyName(String subfamilyName);


    R KinaseGroupOfGroupId(Integer groupId);

    R likeNumber(String number1,String number2);

    R likeActive(String active1,String active2);

    R likeNumberandActive(String number1,String number2,String active1,String active2);
}
