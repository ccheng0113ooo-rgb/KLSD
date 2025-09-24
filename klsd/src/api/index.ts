// import { getBaseURL, get, post, deletes } from "./request";

// const HttpManager = {
//   // // 获取图片信息
//   // attachImageUrl: (url) => url ? `${getBaseURL()}/${url}` : "https://cube.elemecdn.com/e/fd/0fc7d20532fdaf769a25683617711png.png",
//   // // =======================> 用户 API 完成
//   // // 登录
//   // signIn: ({phone,password}) => post(`user/login/status`, {phone,password}),
//   // // 注册
//   // SignUp: ({name, type, sex, phone, identity, password}) => post(`user/add`, {name, type, sex, phone, identity, password}),
//   // // 删除用户
//   // deleteUser: (id) => get(`user/delete?id=${id}`),
//   // // 更新用户信息
//   // updateUserMsg: ({id ,name, type, sex, phone, identity}) => post(`user/update`, {id ,name, type, sex, phone, identity}),
//   // updateUserPassword: ({id, name, oldPassword, password}) => post(`user/updatePassword`, {id, name, oldPassword, password}),
//   // // 返回指定ID的用户
//   // getUserOfId: (id) => get(`user/detail?id=${id}`),
//   // // 更新用户头像
//   // uploadUrl: (userId) => `${getBaseURL()}/user/avatar/update?id=${userId}`,

//   // =======================> chemblgroup API
//   // 获取全部家族
//   getBrowseList: () => get("kinaseGroup"),
//   // 获取group类型
//   getBrowseListOfGroup: (groupname) => get(`kinaseGroup/likeGroupName/detail?groupName=${groupname}`),
//   // 获取subfamily类型
//   getBrowseListOfSubFamily: (subfamilyname) => get(`kinaseGroup/likeSubfamilyName/detail?subfamilyName=${subfamilyname}`),
//   // Number区间
//   //getNumber: (number1, number2) => get(`kinaseGroup/likeNumber/detail?number1=${number1}&&number2=${number2}`),
//   getNumber: (number) => get(`kinaseGroup/likeNumber/detail?${number}`),
//   // Active区间
//   // getActive: (active1, active2) => get(`kinaseGroup/likeActive/detail?active1=${active1}&&active2=${active2}`),
//   getActive: (active) => get(`kinaseGroup/likeActive/detail?${active}`),
//   // Number和Active双区间
//   // getNumberandActive: (number1, number2, active1, active2) => get(`kinaseGroup/likeNumberandActive/detail?number1=${number1}&&number2=${number2}&&active1=${active1}&&active2=${active2}`),
//   getNumberandActive: (numberandactive) => get(`kinaseGroup/likeNumberandActive/detail?${numberandactive}`),
//   // =======================> target API
//   // 计算target1和target2的结果
//   // getTargetList: (targetName1, targetName2) => get(`compoundlist/target/detail?Name1=${targetName1}&&Name2=${targetName2}`),
//   getTargetList: (target) => get(`compoundlist/target/detail?${target}`),


//   // =======================> compound API
//   // 获取全部compound
//   // getSearchList: () => get("compoundlist"),
//   getSearchList: () => get("compoundlist/likeName/detail?Name=jak1"),
//   // 返回包含MoleculechemblId的compound
//   getSearchListOfLikeMoleculechembld: (moleculechemblId) => get(`compoundlist/detail?moleculechemblId=${moleculechemblId}`),
//   // 返回包含name的compound
//   getSearchListOfLikeName: (name) => get(`compoundlist/likeName/detail?Name=${name}`),

//   // =======================> drugs API
//   // 获取全部drugs
//   // getDrugsList: () => get("drugslist"),
//   getDrugsList: () => get("drugslist/likeName/detail?Name=akt"),
//   // 返回包含MoleculechemblId的drugs
//   getDrugsListOfLikeMoleculechembld: (moleculechemblId) => get(`/drugslist/detail?moleculechemblId=${moleculechemblId}`),
//   // 返回包含name的drugs
//   getDrugsListOfLikeName: (name) => get(`drugslist/likeName/detail?Name=${name}`),
//   // 返回包含name的drugs
//   getDrugsListOfLikeDrugName: (drugname) => get(`drugslist/likeDrugsName/detail?drugsName=${drugname}`),

//   // =======================> chemblKinase API
//   // 获取全部激酶
//   // getChemblKinase: () => get("chemblKinase"),
//   getChemblKinase: () => get("chemblKinase/likeTargetName/detail?targetName=jak1"),
//   // 返回包含MoleculechemblId的激酶
//   getChemblKinaseOfLikeMoleculechembld: (moleculechemblId) => get(`chemblKinase/likeMoleculechemblId/detail?moleculechemblId=${moleculechemblId}`),
//   // 返回包含TargetName的激酶
//   getChemblKinaseOfLikeTargetName: (targetName) => get(`chemblKinase/likeTargetName/detail?targetName=${targetName}`),
//     // 返回包含TargetName的激酶
//   getChemblKinaseOfLikePChemblValue: (targetNameandpchemblvalue) => get(`chemblKinaseList/likepchemblvalue/detail?${targetNameandpchemblvalue}`),

//   // =======================> 收藏 API 完成
//   // // 返回的指定用户ID的收藏列表
//   // getCollectionOfUser: (userId) => get(`collection/detail?userId=${userId}`),
//   // // 添加收藏的歌曲 type: 0 代表歌曲， 1 代表歌单
//   // setCollection: ({userId,type,songId}) => post(`collection/add`,{userId,type,songId}),

//   // deleteCollection: (userId, songId) => deletes(`collection/delete?userId=${userId}&&songId=${songId}`),

//   // isCollection: ({userId, type, songId}) => post(`collection/status`, {userId, type, songId}),

// };



// export { HttpManager };





//解决报错
import { getBaseURL, get, post, deletes } from "./request";

const HttpManager = {
  // =======================> chemblgroup API
  // 获取全部家族
  getBrowseList: () => get("kinaseGroup"),
  
  // 获取group类型
  getBrowseListOfGroup: (groupname) => get(`kinaseGroup/likeGroupName/detail?groupName=${encodeURIComponent(groupname)}`),
  
  // 获取subfamily类型
  getBrowseListOfSubFamily: (subfamilyname) => get(`kinaseGroup/likeSubfamilyName/detail?subfamilyName=${encodeURIComponent(subfamilyname)}`),
  
  // Number区间
  getNumber: ({number1, number2}) => get(`kinaseGroup/likeNumber/detail?number1=${number1}&number2=${number2}`),
  
  // Active区间
  getActive: ({active1, active2}) => get(`kinaseGroup/likeActive/detail?active1=${active1}&active2=${active2}`),
  
  // Number和Active双区间
  getNumberandActive: ({number1, number2, active1, active2}) => 
    get(`kinaseGroup/likeNumberandActive/detail?number1=${number1}&number2=${number2}&active1=${active1}&active2=${active2}`),
  
  // =======================> target API
  // 计算target1和target2的结果
  getTargetList: ({Name1, Name2, diff1, diff2}) => 
    get(`compoundlist/target/detail?Name1=${encodeURIComponent(Name1)}&Name2=${encodeURIComponent(Name2)}&diff1=${diff1}&diff2=${diff2}`),

  // =======================> compound API
  // 获取全部compound
  getSearchList: () => get("compoundlist/likeName/detail?Name=jak1"),
  
  // 返回包含MoleculechemblId的compound
  getSearchListOfLikeMoleculechembld: (moleculechemblId) => 
    get(`compoundlist/detail?moleculechemblId=${encodeURIComponent(moleculechemblId)}`),
  
  // 返回包含name的compound
  getSearchListOfLikeName: (name) => 
    get(`compoundlist/likeName/detail?Name=${encodeURIComponent(name)}`),

  // =======================> drugs API
  // 获取全部drugs
  getDrugsList: () => get("drugslist/likeName/detail?Name=akt"),
  
  // 返回包含MoleculechemblId的drugs
  getDrugsListOfLikeMoleculechembld: (moleculechemblId) => 
    get(`/drugslist/detail?moleculechemblId=${encodeURIComponent(moleculechemblId)}`),
  
  // 返回包含name的drugs
  getDrugsListOfLikeName: (name) => 
    get(`drugslist/likeName/detail?Name=${encodeURIComponent(name)}`),
  
  // 返回包含name的drugs
  getDrugsListOfLikeDrugName: (drugname) => 
    get(`drugslist/likeDrugsName/detail?drugsName=${encodeURIComponent(drugname)}`),

  // =======================> chemblKinase API
  // 获取全部激酶
  getChemblKinase: () => get("chemblKinase/likeTargetName/detail?targetName=jak1"),
  
  // 返回包含MoleculechemblId的激酶
  getChemblKinaseOfLikeMoleculechembld: (moleculechemblId) => 
    get(`chemblKinase/likeMoleculechemblId/detail?moleculechemblId=${encodeURIComponent(moleculechemblId)}`),
  
  // 返回包含TargetName的激酶
  getChemblKinaseOfLikeTargetName: (targetName) => 
    get(`chemblKinase/likeTargetName/detail?targetName=${encodeURIComponent(targetName)}`),
  
  // 返回包含TargetName和pChemblValue的激酶
  getChemblKinaseOfLikePChemblValue: ({targetName, pchemblValue}) => 
    get(`chemblKinaseList/likepchemblvalue/detail?targetName=${encodeURIComponent(targetName)}&pchemblValue=${pchemblValue}`)
};

export { HttpManager };