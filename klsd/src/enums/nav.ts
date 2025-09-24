import { RouterName } from "./router-name";

export const enum NavName {
  Home = "Home",
  Browse = "Family",
  Search = "Search",
  Target = "Target",
  Compound = "Compound",
  CompoundPrediction = "Prediction", // 新增
  Drugs = "Drugs",
  Favorites = "Molecule",
  More = "More",
  Faq = "Help",
  DataChart = "DataChart",
  News = "Related Articles",
  ContactUs = "Contact us",
  Personal = "Personal Center",
  Setting = "Settings",
  SignIn = "Login",
  SignUp = "Register",
  SignOut = "Quit",
}

// 左侧导航栏
export const HEADERNAVLIST = [
  {
    name: NavName.Home,
    path: RouterName.Home,
  },
  {
    name: NavName.Browse,
    path: RouterName.Browse,
  },
  {
    name: NavName.Search,
    path: RouterName.Search,
  },
  {
    name: NavName.CompoundPrediction, // 新增
    path: RouterName.CompoundPrediction,
  },
  {
    name: NavName.Drugs,
    path: RouterName.Drugs,
  },
  {
    name: NavName.Favorites,
    path: RouterName.Favorites,
  },
];

export const MORE = [
  {
    name: NavName.More,
  }
];

// 右侧导航栏
export const SIGNLIST = [
  {
    name: NavName.SignIn,
    path: RouterName.SignIn,
  },
  {
    name: NavName.SignUp,
    path: RouterName.SignUp,
  },
];

// 用户下拉菜单项
export const MENULIST = [
  {
    name: NavName.Personal,
    path: RouterName.Personal,
  },
  {
    name: NavName.Setting,
    path: RouterName.Setting,
  },
  {
    name: NavName.SignOut,
    path: RouterName.SignOut,
  },
];

// MORE下拉菜单项
export const MORELIST = [
  {
    name: NavName.News,
    path: RouterName.News,
  },
  {
    name: NavName.DataChart,
    path: RouterName.DataChart,
  },
  {
    name: NavName.Faq,
    path: RouterName.Faq,
  },
  {
    name: NavName.ContactUs,
    path: RouterName.ContactUs,
  }
];