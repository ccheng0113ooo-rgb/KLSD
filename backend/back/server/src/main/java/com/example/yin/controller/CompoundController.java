package com.example.yin.controller;

import com.example.yin.common.R;
import com.example.yin.model.request.CompoundRequest;
import com.example.yin.service.CompoundService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
public class CompoundController {
    @Autowired
    private CompoundService compoundService;


    //TODO 这块就是前端显现相应的compoundlist
    // 返回所有compound
    @GetMapping("/compoundlist")
    public R allCompound() {
        return compoundService.allCompound();
    }

    // 返回moleculechemblId的compound
    @GetMapping("/compoundlist/detail")
    public R compoundOfMoleculeChemblId(@RequestParam String moleculechemblId) {
        return compoundService.CompoundOfMoleculeChemblId(moleculechemblId);
    }

    // 返回包含Name文字的compound
    @GetMapping("/compoundlist/likeName/detail")
    public R compoundOfLikeName(@RequestParam String Name) {
        return compoundService.likeName('%' + Name + '%');
    }

    // 返回包含targetName1和targetName2的compound，并进行计算
    @GetMapping("/compoundlist/target/detail")
    public  R compoundOfTarget(@RequestParam String Name1 , String Name2, String diff1, String diff2) {
        return compoundService.target(Name1,Name2,diff1,diff2);
    }

    // 更新compound信息
    @PostMapping("/compoundlist/update")
    public R updateCompoundMsg(@RequestBody CompoundRequest updateCompoundRequest) {
        return compoundService.updateCompoundMsg(updateCompoundRequest);
    }
}
