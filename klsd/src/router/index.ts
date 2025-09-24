



//预测
// // src/router/index.ts
// import { createRouter, createWebHistory, RouteRecordRaw } from "vue-router";

// const routes: Array<RouteRecordRaw> = [
//   {
//     path: "/:pathMatch(.*)*",
//     redirect: "/404",
//   },
//   {
//     path: "/404",
//     component: () => import("@/views/error/404.vue"),
//   },
//   {
//     path: "/",
//     name: "klsd-container",
//     component: () => import("@/views/KlsdContainer.vue"),
//     children: [
//       {
//         path: "/",
//         name: "home",
//         component: () => import("@/views/Home.vue"),
//       },
//       // {
//       //   path: "/sign-in",
//       //   name: "sign-in",
//       //   component: () => import("@/views/SignIn.vue"),
//       // },
//       // {
//       //   path: "/sign-up",
//       //   name: "sign-up",
//       //   component: () => import("@/views/SignUp.vue"),
//       // },
//       // {
//       //   path: "/personal",
//       //   name: "personal",
//       //   meta: {
//       //     requireAuth: true,
//       //   },
//       //   component: () => import("@/views/personal/Personal.vue"),
//       // },
//       {
//         path: "/browse",
//         name: "browse",
//         component: () => import("@/views/browse/Browse.vue"),
//       },
//       // {
//       //   path: "/browse-detail/:id",
//       //   name: "browse-detail",
//       //   component: () => import("@/views/browse/BrowseDetail.vue"),
//       // },
//       {
//         path: "/drugs",
//         name: "drugs",
//         component: () => import("@/views/drugs/Drugs.vue"),
//       },
//       // {
//       //   path: "/drugs/:id",
//       //   name: "drugs-detail",
//       //   component: () => import("@/views/drugs/DrugsDetail.vue"),
//       // },
//       {
//         path: "/favorites",
//         name: "favorites",
//         component: () => import("@/views/favorites/Favorites.vue"),
//       },
//       // {
//       //   path: "/favorites/:id",
//       //   name: "favorites-detail",
//       //   component: () => import("@/views/favorites/FavoritesDetail.vue"),
//       // },
//       {
//         path: "/target",
//         name: "target",
//         component: () => import("@/views/search/Target.vue"),
//       },
//       {
//         path: "/compound",
//         name: "compound",
//         component: () => import("@/views/search/Compound.vue"),
//       },
//       {
//         path: "/compound-prediction",
//         name: "compound-prediction",
//         component: () => import("@/views/search/CompoundPrediction.vue"),
//         meta: {
//           title: "化合物活性预测"
//         }
//       },
//       // {
//       //   path: "/personal-data",
//       //   name: "personal-data",
//       //   component: () => import("@/views/setting/PersonalData.vue"),
//       // },
//       {
//         path: "/faq",
//         name: "faq",
//         component: () => import("@/views/more/Faq.vue"),
//       },
//       {
//         path: "/datachart",
//         name: "datachart",
//         component: () => import("@/views/more/DataChart.vue"),
//       },
//       {
//         path: "/news",
//         name: "news",
//         component: () => import("@/views/more/News.vue"),
//       },
//       {
//         path: "/contactus",
//         name: "contactus",
//         component: () => import("@/views/more/ContactUs.vue"),
//       },
//       // {
//       //   path: "/setting",
//       //   name: "setting",
//       //   meta: {
//       //     requireAuth: true,
//       //   },
//       //   component: () => import("@/views/setting/Setting.vue"),
//       //   children: [
//       //     {
//       //       path: "/setting/PersonalData",
//       //       name: "personalData",
//       //       meta: {
//       //         requireAuth: true,
//       //       },
//       //       component: () => import("@/views/setting/PersonalData.vue"),
//       //     }
//       //   ]
//       // },
//     ],
//   },
// ];

// const router = createRouter({
//   history: createWebHistory(process.env.BASE_URL),
//   routes,
// });

// // 设置页面标题
// router.beforeEach((to, from, next) => {
//   const defaultTitle = "你的网站名称";
//   document.title = to.meta.title ? `${to.meta.title} | ${defaultTitle}` : defaultTitle;
//   next();
// });

// export default router;





// src/router/index.ts
import { createRouter, createWebHistory, RouteRecordRaw } from "vue-router";

const routes: Array<RouteRecordRaw> = [
  {
    path: "/:pathMatch(.*)*",
    redirect: "/404",
  },
  {
    path: "/404",
    component: () => import("@/views/error/404.vue"),
  },
  {
    path: "/",
    name: "klsd-container",
    component: () => import("@/views/KlsdContainer.vue"),
    children: [
      {
        path: "/",
        name: "home",
        component: () => import("@/views/Home.vue"),
        meta: {
          title: "首页"
        }
      },
      {
        path: "/browse",
        name: "browse",
        component: () => import("@/views/browse/Browse.vue"),
      },
      {
        path: "/drugs",
        name: "drugs",
        component: () => import("@/views/drugs/Drugs.vue"),
      },
      {
        path: "/favorites",
        name: "favorites",
        component: () => import("@/views/favorites/Favorites.vue"),
      },
      {
        path: "/target",
        name: "target",
        component: () => import("@/views/search/Target.vue"),
      },
      {
        path: "/compound",
        name: "compound",
        component: () => import("@/views/search/Compound.vue"),
      },
      {
        path: "/compound-prediction",
        name: "compound-prediction",
        component: () => import("@/views/search/CompoundPrediction.vue"),
        meta: {
          title: "化合物活性预测"   
      },
      props: (route) => ({
        smiles: route.query.smiles || ''
      })
    },
      // 新增带参数的路由 - 方案1
      {
        path: "/compound-prediction/:smiles",
        name: "compound-prediction-with-param",
        component: () => import("@/views/search/CompoundPrediction.vue"),
        meta: {
          title: "化合物活性预测"
        },
        props: true // 启用props接收参数
      },
      // 或者方案2 - 使用query参数（不需要额外路由）
      // 现有的compound-prediction路由已经可以处理query参数
      {
        path: "/faq",
        name: "faq",
        component: () => import("@/views/more/Faq.vue"),
      },
      {
        path: "/datachart",
        name: "datachart",
        component: () => import("@/views/more/DataChart.vue"),
      },
      {
        path: "/news",
        name: "news",
        component: () => import("@/views/more/News.vue"),
      },
      {
        path: "/contactus",
        name: "contactus",
        component: () => import("@/views/more/ContactUs.vue"),
      },
    ],
  },
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
});

// 设置页面标题
router.beforeEach((to, from, next) => {
  const defaultTitle = "KLSD";
  document.title = to.meta.title ? `${to.meta.title} | ${defaultTitle}` : defaultTitle;
  next();
});

export default router;