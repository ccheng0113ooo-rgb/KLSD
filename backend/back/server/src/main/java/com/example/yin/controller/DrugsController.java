package com.example.yin.controller;

import com.example.yin.common.R;
import com.example.yin.model.request.DrugsRequest;
import com.example.yin.service.DrugsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
public class DrugsController {
    @Autowired
    private DrugsService drugsService;


    //TODO 这块就是前端显现相应的drugslist
    // 返回所有drugs
    @GetMapping("/drugslist")
    public R allDrugs() {
        return drugsService.allDrugs();
    }

    // 返回moleculechemblId的drugs
    @GetMapping("/drugslist/detail")
    public R drugsOfMoleculeChemblId(@RequestParam String moleculechemblId) {
        return drugsService.DrugsOfMoleculeChemblId(moleculechemblId);
    }

    // 返回包含Name文字的drugs
    @GetMapping("/drugslist/likeName/detail")
    public R drugsOfLikeName(@RequestParam String Name) {
        return drugsService.likeName('%' + Name + '%');
    }

    // 返回包含Drugs_Name文字的drugs
    @GetMapping("/drugslist/likeDrugsName/detail")
    public R drugsOfLikeDrugsName(@RequestParam String drugsName) {
        return drugsService.likeDrugsName('%' + drugsName + '%');
    }

    // 更新drugs
    @PostMapping("/drugslist/update")
    public R updateDrugsMsg(@RequestBody DrugsRequest updateDrugsRequest) {
        return drugsService.updateDrugsMsg(updateDrugsRequest);
    }
}
