package com.example.yin.controller;

import com.example.yin.common.R;
import com.example.yin.model.request.KinaseGroupRequest;
import com.example.yin.service.KinaseGroupService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
public class KinaseGroupController {

    @Autowired
    private KinaseGroupService kinaseGroupService;


    //TODO 这块就是前端显现相应的激酶家族list
    // 返回所有激酶
    @GetMapping("/kinaseGroup")
    public R allKinaseGroup() {
        return kinaseGroupService.allKinaseGroup();
    }

    // 返回groupId的激酶家族
    @GetMapping("/kinaseGroup/detail")
    public R KinaseGroupOfGroupId(@RequestParam int groupId) {
        return kinaseGroupService.KinaseGroupOfGroupId(groupId);
    }


    // 返回包含groupName文字的激酶家族
    @GetMapping("/kinaseGroup/likeGroupName/detail")
    public R kinaseGroupOfLikeGroupName(@RequestParam String groupName) {
        return kinaseGroupService.likeGroupName('%' + groupName + '%');
    }

    // 返回包含subfamilyname文字的激酶家族
    @GetMapping("/kinaseGroup/likeSubfamilyName/detail")
    public R kinaseGroupOfLikeSubfamilyName(@RequestParam String subfamilyName) {
        return kinaseGroupService.likeSubfamilyName('%' + subfamilyName + '%');
    }

    // 返回number区间的激酶家族
    @GetMapping("/kinaseGroup/likeNumber/detail")
    public R kinaseGroupOfLikeNumber(@RequestParam String number1, String number2) {
        return kinaseGroupService.likeNumber(number1,number2);
    }

    // 返回active区间的激酶家族
    @GetMapping("/kinaseGroup/likeActive/detail")
    public R kinaseGroupOfLikeActive(@RequestParam String active1, String active2) {
        return kinaseGroupService.likeActive(active1,active2);
    }

    // 返回numberhe和active区间的激酶家族
    @GetMapping("/kinaseGroup/likeNumberandActive/detail")
    public R kinaseGroupOfLikeNumberandActive(@RequestParam String number1, String number2, String active1, String active2) {
        return kinaseGroupService.likeNumberandActive(number1,number2,active1,active2);
    }

    // 更新激酶家族信息
    @PostMapping("/kinaseGroupList/update")
    public R updateKinaseGroupMsg(@RequestBody KinaseGroupRequest updateKinaseGroupRequest) {
        return kinaseGroupService.updateKinaseGroupMsg(updateKinaseGroupRequest);
    }
}
