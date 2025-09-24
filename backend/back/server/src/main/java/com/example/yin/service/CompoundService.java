package com.example.yin.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.example.yin.common.R;
import com.example.yin.model.domain.Compound;
import com.example.yin.model.request.CompoundRequest;

import java.util.List;
import java.util.Map;

public interface CompoundService extends IService<Compound> {
    R updateCompoundMsg(CompoundRequest updateCompoundRequest);

    R allCompound();

    R likeName(String Name);

    R CompoundOfMoleculeChemblId(String moleculechemblId);

    R target(String Name1, String Name2, String diff1, String diff2);
}
