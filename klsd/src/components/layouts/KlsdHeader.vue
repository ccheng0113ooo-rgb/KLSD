<template>
  <div class="klsd-header">
    <!--图标-->
    <div class="header-logo" @click="goPage()">
      <span>{{ databaseName }}</span>
    </div>
    <klsd-header-nav class="klsd-header-nav1" :styleList="headerNavList" :activeName="activeNavName" @click="goPage"></klsd-header-nav>
    <el-dropdown class="more-wrap" trigger="click">
      <klsd-header-nav :styleList="more" :activeName="activeNavName"></klsd-header-nav>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item v-for="(item, index) in moreList" :key="index" @click.stop="goMoreList(item.path)">{{ item.name }}</el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
    <!--设置-->
    <!-- <klsd-header-nav class="klsd-header-nav2" v-if="!token" :styleList="signList" :activeName="activeNavName" @click="goPage"></klsd-header-nav>
    <el-dropdown class="user-wrap" trigger="click">
      <span class="el-dropdown-link">Personal Center</span>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item v-for="(item, index) in menuList" :key="index" @click.stop="goMenuList(item.path)">{{ item.name }}</el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown> -->
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, getCurrentInstance, computed, reactive } from "vue";
import { Search } from "@element-plus/icons-vue";
import { useStore } from "vuex";
import KlsdHeaderNav from "@/components/layouts/KlsdHeaderNav.vue";
import mixin from "@/mixins/mixin";
import { HEADERNAVLIST, SIGNLIST, MENULIST, DATABASENAME, RouterName, NavName ,MORELIST, MORE} from "@/enums";
import { HttpManager } from "@/api";

export default defineComponent({
  data() {
    return {
      showSubMenu: true, // 默认隐藏子菜单
    };
  },
  components: {
    KlsdHeaderNav,
  },
  computed: {
    // hasMoreSubMenu() {
    //   return this.moreList.length > 0; // 判断是否有子菜单数据
    // },
  },
  setup() {
    const { proxy } = getCurrentInstance();
    const store = useStore();
    const { changeIndex, routerManager } = mixin();

    const databaseName = ref(DATABASENAME);
    const headerNavList = ref(HEADERNAVLIST); // 左侧导航栏
    const signList = ref(SIGNLIST); // 右侧导航栏
    const menuList = ref(MENULIST); // 用户下拉菜单项
    const moreList = ref(MORELIST); // MORE下拉菜单项
    const more = ref(MORE); // MORE下拉菜单项
    const keywords = ref("");
    const activeNavName = computed(() => store.getters.activeNavName);
    const token = computed(() => store.getters.token);

    function goPage(path, name) {
      if (!path && !name) {
        changeIndex(NavName.Home);
        routerManager(RouterName.Home, { path: RouterName.Home });
      } else {
        changeIndex(name);
        routerManager(path, { path });
      }
    }

    function goMenuList(path) {
      if (path == RouterName.SignOut) {
        proxy.$store.commit("setToken", false);
        changeIndex(NavName.Home);
        routerManager(RouterName.Home, { path: RouterName.Home });
      } else {
        routerManager(path, { path });
      }
    }

    function goMoreList(path) {
        changeIndex(NavName.More);
        routerManager(RouterName.More, { path: RouterName.More });
        routerManager(path, { path });
    }

    return {
      databaseName,
      headerNavList,
      signList,
      menuList,
      moreList,
      keywords,
      activeNavName,
      token,
      more,
      Search,
      goPage,
      goMenuList,
      goMoreList,
      // attachImageUrl: HttpManager.attachImageUrl,
    };
  },
});
</script>

<style lang="scss" scoped>
@import "@/assets/css/var.scss";
@import "@/assets/css/global.scss";

@media screen and (min-width: $sm) {
  .header-logo {
    margin: 0 1rem;
  }
}

@media screen and (max-width: $sm) {
  .header-logo {
    margin: 0 1rem;
    span {
      display: none;
    }
  }
  .header-search {
    display: none;
  }
}

.klsd-header {
  position: fixed;
  width: 100%;
  height: $header-height;
  line-height: $header-height;
  padding: $header-padding;
  margin: $header-margin;
  background-color: $theme-header-color;
  box-shadow: $box-shadow;
  box-sizing: border-box;
  z-index: 100;
  display: flex;
  white-space: nowrap;
  flex-wrap: nowrap;
}

/* LOGO */
.header-logo {
  font-size: $font-size-logo;
  font-weight: bold;
  cursor: pointer;
  .icon {
    @include icon(1.9rem, $color-black);
    vertical-align: middle;
  }
  span {
    margin-left: 1rem;
  }
}

.klsd-header-nav1 {
  flex: 1;
}

.klsd-header-nav2 {
  flex: 1;
}


/*用户*/
.user-wrap {
  position: relative;
  display: flex;
  align-items: center;

  .user {
    width: $header-user-width;
    height: $header-user-width;
    border-radius: $header-user-radius;
    margin-right: $header-user-margin;
    cursor: pointer;
  }
}

/*more*/
.more-wrap {
  position: relative;
  display: flex;
  align-items: center;
  width: 70%;
  margin-left: 10px;

  .more {
    width: $header-user-width;
    height: $header-user-width;
    border-radius: $header-user-radius;
    margin-right: $header-user-margin;
    margin: $header-nav-margin;
    padding: $header-nav-padding;
    line-height: 3.3rem;
    color: $color-grey;
    border-bottom: none;
    color: $color-black;
    font-weight: 600;
    border-bottom: 5px solid $color-black;
    cursor: pointer;
  }
}

</style>
