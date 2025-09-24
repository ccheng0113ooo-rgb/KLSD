package com.example.yin.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.yin.common.R;
import com.example.yin.mapper.ChemblKinaseMapper;
import com.example.yin.model.domain.ChemblKinase;
import com.example.yin.model.domain.KinaseGroup;
import com.example.yin.model.request.ChemblKinaseRequest;
import com.example.yin.service.ChemblKinaseService;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;

@Service
public class ChemblKinaseServiceImpl extends ServiceImpl<ChemblKinaseMapper, ChemblKinase> implements ChemblKinaseService {

    @Autowired
    private ChemblKinaseMapper chemblKinaseMapper;

    @Override
    public R updateChemblKinaseMsg(ChemblKinaseRequest updateChemblKinaseRequest) {
        return null;
    }

    @Override
    public R likeMoleculechemblId(String moleculechemblId) {
        QueryWrapper<ChemblKinase> queryWrapper = new QueryWrapper<>();
        queryWrapper.like("moleculechemblid",moleculechemblId);
        return R.success(null, chemblKinaseMapper.selectList(queryWrapper));
    }

    @Override
    public R likeTargetName(String targetName) {
        QueryWrapper<ChemblKinase> queryWrapper = new QueryWrapper<>();
        queryWrapper.like("targetname",targetName);
        return R.success(null, chemblKinaseMapper.selectList(queryWrapper));
    }

    @Override
    public R allChemblKinase() {
        return R.success(null, chemblKinaseMapper.selectList(null));
    }

    @Override
    public R likepchemblvalue(String Name1,String pchemblvalue1,String pchemblvalue2){
        QueryWrapper<ChemblKinase> queryWrapper = new QueryWrapper<>();
        queryWrapper.like("targetname",Name1);
        if (pchemblvalue1 != null && pchemblvalue2 != null) {
            // pchemblvalue1和pchemblvalue2之间的数据
            queryWrapper.between("pchemblvalue", pchemblvalue1, pchemblvalue2);
        } else if (pchemblvalue1 != null) {
            // 大于pchemblvalue1的数据
            queryWrapper.ge("pchemblvalue", pchemblvalue1);
        } else if (pchemblvalue2 != null) {
            // 小于pchemblvalue2的数据
            queryWrapper.le("pchemblvalue", pchemblvalue2);
        }
        return R.success(null, chemblKinaseMapper.selectList(queryWrapper));
    }
}
