package com.example.yin.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.yin.common.R;
import com.example.yin.mapper.CompoundMapper;
import com.example.yin.model.domain.Compound;
import com.example.yin.model.domain.Target;
import com.example.yin.model.request.CompoundRequest;
import com.example.yin.service.CompoundService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class CompoundServiceImpl extends ServiceImpl<CompoundMapper, Compound> implements CompoundService {
    @Autowired
    private CompoundMapper compoundMapper;

    @Override
    public R updateCompoundMsg(CompoundRequest updateCompoundRequest) {
        return null;
    }

    @Override
    public R CompoundOfMoleculeChemblId(String moleculechemblIdId) {
        QueryWrapper<Compound> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("molecule_chembl_id",moleculechemblIdId);
        return R.success("查询成功", compoundMapper.selectList(queryWrapper));
    }

    @Override
    public R likeName(String Name) {
        QueryWrapper<Compound> queryWrapper = new QueryWrapper<>();
        queryWrapper.like("name",Name);
        return R.success(null, compoundMapper.selectList(queryWrapper));
    }

    public R target(String Name1, String Name2, String diff1, String diff2) {
        // 将 Name1 和 Name2 转换为小写，用于不区分大小写的查询
        String name1LowerCase = Name1.toLowerCase();
        String name2LowerCase = Name2.toLowerCase();
        QueryWrapper<Compound> queryWrapper = new QueryWrapper<>();
        queryWrapper.apply("LOWER(name) LIKE CONCAT('%', LOWER({0}), '%')", name1LowerCase)
                .and(wrapper -> wrapper.in("standard_type", "IC50", "EC50", "Kd", "Ki","pact")
                        .isNotNull("standard_value")
                        .and(w -> w.ne("standard_value", "").ne("standard_value", "null")))
                .or()
                .apply("LOWER(name) LIKE CONCAT('%', LOWER({0}), '%')", name2LowerCase)
                .and(wrapper -> wrapper.in("standard_type", "IC50", "EC50", "Kd", "Ki","pact")
                        .isNotNull("standard_value")
                        .and(w -> w.ne("standard_value", "").ne("standard_value", "null")));

        List<Compound> compoundEntities = compoundMapper.selectList(queryWrapper);

//        List<Map<String, Object>> result = new ArrayList<>();

        // Map to store targetName records
        Map<String, Compound> targetName1Records = new HashMap<>();
        Map<String, Compound> targetName2Records = new HashMap<>();


        for (Compound entity : compoundEntities) {
            if (entity.getName().toLowerCase().contains(Name1.toLowerCase())) {
                targetName1Records.put(entity.getMoleculeChemblId(), entity);
            } else if (entity.getName().toLowerCase().contains(Name2.toLowerCase())) {
                targetName2Records.put(entity.getMoleculeChemblId(), entity);
            }
        }

        // Iterate through targetName1 records and calculate differences
        Map<String, Target> result = new HashMap<>();
        List<Target> targetList = new ArrayList<>();

        // Iterate over the moleculeChemblIds in targetName1Records
        for (String moleculeChemblId : targetName1Records.keySet()) {
            // Check if the same moleculeChemblId exists in targetName2Records
            if (targetName2Records.containsKey(moleculeChemblId)) {
                Compound compound1 = targetName1Records.get(moleculeChemblId);
                Compound compound2 = targetName2Records.get(moleculeChemblId);

                // Check if the standard type is one of the specified types
                if (isSupportedStandardType(compound1.getStandardType()) &&
                        isSupportedStandardType(compound2.getStandardType())) {
                    Target target = new Target();
                    double standardValue1 = Double.parseDouble(compound1.getStandardValue());
                    double standardValue2 = Double.parseDouble(compound2.getStandardValue());
                    String name1 = compound1.getName();
                    String name2 = compound2.getName();
                    String type1 = compound1.getStandardType();
                    String type2 = compound2.getStandardType();
                    String documentChemblId1 = compound1.getDocumentChemblId();
                    String documentChemblId2 = compound2.getDocumentChemblId();
                    double difference;
                    double pact1;
                    double pact2;

                    // Calculate the difference based on standard type
                    if(type1.equals("pact")){
                        pact1 = standardValue1;
                    }else{
                        pact1 = 9 - Math.log10(standardValue1);
                    }
                    if(type2.equals("pact")){
                        pact2 = standardValue2;
                    }else{
                        pact2 = 9 - Math.log10(standardValue2);
                    }
                    difference = pact1 - pact2;
                    if (diff1 == null && diff2 == null){
                        target.setMoleculeChemblId(moleculeChemblId);
                        target.setTargetname1(name1);
                        target.setTargetname2(name2);
                        target.setDocumentChemblId1(documentChemblId1);
                        target.setDocumentChemblId2(documentChemblId2);
                        target.setDiff(difference);
                        target.setPact1(pact1);
                        target.setPact2(pact2);
                        targetList.add(target);
                        // Put the difference into the result map with the moleculeChemblId as key
                        result.put(moleculeChemblId,target);
                    }else if(diff1 != null && diff2 == null){
                        double min = Double.parseDouble(diff1);
                        if(difference >= min){
                            target.setMoleculeChemblId(moleculeChemblId);
                            target.setTargetname1(name1);
                            target.setTargetname2(name2);
                            target.setDocumentChemblId1(documentChemblId1);
                            target.setDocumentChemblId2(documentChemblId2);
                            target.setDiff(difference);
                            target.setPact1(pact1);
                            target.setPact2(pact2);
                            targetList.add(target);
                            // Put the difference into the result map with the moleculeChemblId as key
                            result.put(moleculeChemblId,target);
                        }
                    }else if(diff1 == null && diff2 != null){
                        double max = Double.parseDouble(diff2);
                        if(difference <= max){
                            target.setMoleculeChemblId(moleculeChemblId);
                            target.setTargetname1(name1);
                            target.setTargetname2(name2);
                            target.setDocumentChemblId1(documentChemblId1);
                            target.setDocumentChemblId2(documentChemblId2);
                            target.setDiff(difference);
                            target.setPact1(pact1);
                            target.setPact2(pact2);
                            targetList.add(target);
                            // Put the difference into the result map with the moleculeChemblId as key
                            result.put(moleculeChemblId,target);
                        }
                    }else{
                        double min = Double.parseDouble(diff1);
                        double max = Double.parseDouble(diff2);
                        if(difference >= min&&difference <= max){
                            target.setMoleculeChemblId(moleculeChemblId);
                            target.setTargetname1(name1);
                            target.setTargetname2(name2);
                            target.setDocumentChemblId1(documentChemblId1);
                            target.setDocumentChemblId2(documentChemblId2);
                            target.setDiff(difference);
                            target.setPact1(pact1);
                            target.setPact2(pact2);
                            targetList.add(target);
                            // Put the difference into the result map with the moleculeChemblId as key
                            result.put(moleculeChemblId,target);
                        }
                    }
                }
            }
        }

        return R.success("查询成功",targetList);
        }
    private static boolean isSupportedStandardType(String standardType) {
        return standardType.equals("IC50") ||
                standardType.equals("EC50") ||
                standardType.equals("Kd") ||
                standardType.equals("Ki");
    }
    @Override
    public R allCompound() {
        return R.success(null, compoundMapper.selectList(null));
    }
}
