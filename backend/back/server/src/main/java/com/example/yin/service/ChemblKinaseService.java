package com.example.yin.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.example.yin.common.R;
import com.example.yin.model.domain.ChemblKinase;
import com.example.yin.model.request.ChemblKinaseRequest;
import org.springframework.web.multipart.MultipartFile;

public interface ChemblKinaseService extends IService<ChemblKinase> {

    R updateChemblKinaseMsg(ChemblKinaseRequest updateChemblKinaseRequest);

    R allChemblKinase();

    R likeMoleculechemblId(String moleculechemblid);

    R likeTargetName(String targetName);

    R likepchemblvalue(String Name1,String pchemblvalue1,String pchemblvalue2);
}
