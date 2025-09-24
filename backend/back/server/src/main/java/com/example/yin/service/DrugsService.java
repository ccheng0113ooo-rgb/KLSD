package com.example.yin.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.example.yin.common.R;
import com.example.yin.model.domain.Drugs;
import com.example.yin.model.request.DrugsRequest;

public interface DrugsService extends IService<Drugs> {

    R updateDrugsMsg(DrugsRequest updateDrugsRequest);

    R allDrugs();

    R likeName(String Name);

    R likeDrugsName(String drugsName);

    R DrugsOfMoleculeChemblId(String moleculechemblId);

}
