import { getCurrentInstance, computed } from "vue";
import { useStore } from "vuex";
import { LocationQueryRaw } from "vue-router";
import { RouterName } from "@/enums";
import { HttpManager } from "@/api";

interface routerOptions {
  path?: string;
  query?: LocationQueryRaw;
}

export default function () {
  const { proxy } = getCurrentInstance();

  const store = useStore();
  const token = computed(() => store.getters.token);

  function getUserSex(sex) {
    if (sex === 0) {
      return "women";
    } else if (sex === 1) {
      return "men";
    }
  }

  // 获取Browse列表
  function getBrowseList(str) {
    return str.split("-")[1];
  }
  
  // 获取Search列表
  function getSearchList(str) {
    return str.split("-")[1];
  }

  // 获取Drugs列表
  function getDrugsList(str) {
    return str.split("-")[1];
  }
  
  // 获取Favorites列表
  function getFavoritesList(str) {
    return str.split("-")[1];
  }

  // 获取FAQ列表
  function getFAQList(str) {
    return str.split("-")[1];
  }

  // 获取News列表
  function getNewsList(str) {
    return str.split("-")[1];
  }

  // 判断登录状态
  function checkStatus(status?: boolean) {
    if (!token.value) {
      if (status !== false)
        (proxy as any).$message({
          message: "Please login first.",
          type: "warning",
        });
      return false;
    }
    return true;
  }

  // 导航索引
  function changeIndex(value) {
    proxy.$store.commit("setActiveNavName", value);
  }
  // 路由管理
  function routerManager(routerName: string | number, options: routerOptions) {
    switch (routerName) {
      // case RouterName.Search:
      //   proxy.$router.push({ path: options.path, query: options.query });
      //   break;
      case RouterName.Home:
      case RouterName.Browse:
      case RouterName.BrowseDetail:
      case RouterName.Target: 
      case RouterName.Compound:  
      case RouterName.SearchDetail:
      case RouterName.Drugs:
      case RouterName.DrugsDetail:
      case RouterName.Favorites:
      case RouterName.FavoritesDetail:
      case RouterName.More:
      case RouterName.MoreDetail:
      case RouterName.Faq:
      case RouterName.DataChart:
      case RouterName.News:
      case RouterName.ContactUs:
      case RouterName.Personal:
      case RouterName.PersonalData:
      case RouterName.Setting:
      case RouterName.SignIn:
      case RouterName.SignUp:
      case RouterName.SignOut:
      case RouterName.Error:
      default:
        proxy.$router.push({ path: options.path });
        break;
    }
  }

  function goBack(step = -1) {
    proxy.$router.go(step);
  }

  return {
    getUserSex,
    getBrowseList,
    getSearchList,
    getDrugsList,
    getFavoritesList,
    getFAQList,
    getNewsList,
    changeIndex,
    checkStatus,
    routerManager,
    goBack,
  };
}
