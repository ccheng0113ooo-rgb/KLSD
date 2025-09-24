package com.example.yin.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.example.yin.common.R;
import com.example.yin.model.domain.ChemblKinase;
import com.example.yin.model.request.ChemblKinaseRequest;
import com.example.yin.service.ChemblKinaseService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
public class ChemblKinaseController {

    @Autowired
    private ChemblKinaseService chemblKinaseService;


    //TODO 这块就是前端显现相应的激酶list
    // 返回所有激酶
    @GetMapping("/chemblKinase")
    public R allChemblKinase() {
        return chemblKinaseService.allChemblKinase();
    }

    // 分页返回所有激酶
    @GetMapping("/chemblKinaseList")
    public Page<ChemblKinase> getEntities(@RequestParam(defaultValue = "1") long current,
                                          @RequestParam(defaultValue = "50") long size) {
        Page<ChemblKinase> page = new Page<>(current, size);
        return chemblKinaseService.page(page);
    }

    // 返回pchemblvalue区间的激酶家族
    @GetMapping("/chemblKinaseList/likepchemblvalue/detail")
    public R chemblKinaseOfLikepchemblvalue(@RequestParam String Name1,String pchemblvalue1, String pchemblvalue2) {
        return chemblKinaseService.likepchemblvalue(Name1, pchemblvalue1,pchemblvalue2);
    }

    // 返回chemblId包含文字的激酶
    @GetMapping("/chemblKinase/likeMoleculechemblId/detail")
    public R chemblKinaseOfLikeMoleculechemblId(@RequestParam String moleculechemblId) {
        return chemblKinaseService.likeMoleculechemblId('%' + moleculechemblId + '%');
    }

    // 返回包含TargetName的激酶
    @GetMapping("/chemblKinase/likeTargetName/detail")
    public R chemblKinaseOfLikeTargetName(@RequestParam String targetName) {
        return chemblKinaseService.likeTargetName('%' + targetName + '%');
    }

    // 更新激酶信息
    @PostMapping("/chemblKinaseList/update")
    public R updateChemblKinaseMsg(@RequestBody ChemblKinaseRequest updateChemblKinaseRequest) {
        return chemblKinaseService.updateChemblKinaseMsg(updateChemblKinaseRequest);

    }
}
