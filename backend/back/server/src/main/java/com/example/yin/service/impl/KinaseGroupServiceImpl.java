package com.example.yin.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.yin.common.R;
import com.example.yin.mapper.KinaseGroupMapper;
import com.example.yin.model.domain.KinaseGroup;
import com.example.yin.model.request.KinaseGroupRequest;
import com.example.yin.service.KinaseGroupService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class KinaseGroupServiceImpl extends ServiceImpl<KinaseGroupMapper, KinaseGroup> implements KinaseGroupService {

    @Autowired
    private KinaseGroupMapper kinaseGroupMapper;

    @Override
    public R updateKinaseGroupMsg(KinaseGroupRequest updateKinaseGroupRequest) {
        return null;
    }

    @Override
    public R KinaseGroupOfGroupId(Integer groupId) {
        QueryWrapper<KinaseGroup> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("groupid",groupId);
        return R.success("查询成功", kinaseGroupMapper.selectList(queryWrapper));
    }

    @Override
    public R likeGroupName(String groupName) {
        QueryWrapper<KinaseGroup> queryWrapper = new QueryWrapper<>();
        queryWrapper.like("groupname",groupName);
        return R.success(null, kinaseGroupMapper.selectList(queryWrapper));
    }

    @Override
    public R likeSubfamilyName(String subfamliyName) {
        QueryWrapper<KinaseGroup> queryWrapper = new QueryWrapper<>();
        queryWrapper.like("subfamilyname",subfamliyName);
        return R.success(null, kinaseGroupMapper.selectList(queryWrapper));
    }

    @Override
    public R allKinaseGroup() {
        return R.success(null, kinaseGroupMapper.selectList(null));
    }

    @Override
    public R likeNumber(String number1,String number2) {
        QueryWrapper<KinaseGroup> queryWrapper = new QueryWrapper<>();
        // 添加数值范围查询条件
        if (number1 != null && number2 != null) {
            queryWrapper.between("number", number1, number2); // 替换 "number_column" 为实际数据库表中的列名
        } else if (number1 != null) {
            queryWrapper.ge("number", number1); // 只有 number1 的条件
        } else if (number2 != null) {
            queryWrapper.le("number", number2); // 只有 number2 的条件
        }
        return R.success(null, kinaseGroupMapper.selectList(queryWrapper));
    }

    @Override
    public R likeActive(String active1, String active2) {
        QueryWrapper<KinaseGroup> queryWrapper = new QueryWrapper<>();

        if (active1 != null && active2 != null) {
            // active1和active2之间的数据
            queryWrapper.between("active", active1, active2);
        } else if (active1 != null) {
            // 大于active1的数据
            queryWrapper.ge("active", active1);
        } else if (active2 != null) {
            // 小于active2的数据
            queryWrapper.le("active", active2);
        }
        return R.success(null, kinaseGroupMapper.selectList(queryWrapper));
    }

    @Override
    public R likeNumberandActive(String number1,String number2,String active1,String active2) {
        QueryWrapper<KinaseGroup> queryWrapper = new QueryWrapper<>();
        // 添加number数值范围查询条件
        if (number1 != null && number2 != null) {
            queryWrapper.between("number", number1, number2); // 替换 "number_column" 为实际数据库表中的列名
        } else if (number1 != null) {
            queryWrapper.ge("number", number1); // 只有 number1 的条件
        } else if (number2 != null) {
            queryWrapper.le("number", number2); // 只有 number2 的条件
        }
        // 添加active数值范围查询条件
        if (active1 != null && active2 != null) {
            // active1和active2之间的数据
            queryWrapper.between("active", active1, active2);
        } else if (active1 != null) {
            // 大于active1的数据
            queryWrapper.ge("active", active1);
        } else if (active2 != null) {
            // 小于active2的数据
            queryWrapper.le("active", active2);
        }
        return R.success(null, kinaseGroupMapper.selectList(queryWrapper));
    }
}
