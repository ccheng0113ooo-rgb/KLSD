package com.example.yin.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.yin.common.R;
import com.example.yin.mapper.DrugsMapper;
import com.example.yin.model.domain.Drugs;
import com.example.yin.model.request.DrugsRequest;
import com.example.yin.service.DrugsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class DrugsServiceImpl extends ServiceImpl<DrugsMapper, Drugs> implements DrugsService {
    @Autowired
    private DrugsMapper drugsMapper;

    @Override
    public R updateDrugsMsg(DrugsRequest updateDrugsRequest) {
        return null;
    }

    @Override
    public R DrugsOfMoleculeChemblId(String moleculechemblIdId) {
        QueryWrapper<Drugs> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("molecule_chembl_id",moleculechemblIdId);
        return R.success("查询成功",drugsMapper.selectList(queryWrapper));
    }

    @Override
    public R likeName(String Name) {
        QueryWrapper<Drugs> queryWrapper = new QueryWrapper<>();
        queryWrapper.like("name",Name);
        return R.success(null, drugsMapper.selectList(queryWrapper));
    }

    @Override
    public R likeDrugsName(String drugsName) {
        QueryWrapper<Drugs> queryWrapper = new QueryWrapper<>();
        queryWrapper.like("drug_name",drugsName);
        return R.success(null, drugsMapper.selectList(queryWrapper));
    }

    @Override
    public R allDrugs() {
        return R.success(null, drugsMapper.selectList(null));
    }
}
