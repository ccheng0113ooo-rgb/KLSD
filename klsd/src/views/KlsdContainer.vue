<template>
  <el-container>
    <el-header>
      <klsd-header></klsd-header>
    </el-header>
    <el-main>
      <router-view />
      <klsd-scroll-top></klsd-scroll-top>
    </el-main>
    <el-footer>
      <klsd-footer></klsd-footer>
    </el-footer>
  </el-container>
</template>

<script lang="ts">
import {getCurrentInstance } from "vue";
import KlsdHeader from "@/components/layouts/KlsdHeader.vue";
import KlsdScrollTop from "@/components/layouts/KlsdScrollTop.vue";
import KlsdFooter from "@/components/layouts/KlsdFooter.vue";

export default {
  components: {
    KlsdHeader,
    KlsdScrollTop,
    KlsdFooter,
  },

  setup () {
    const { proxy } = getCurrentInstance();

    if (sessionStorage.getItem("dataStore")) {
      proxy.$store.replaceState(Object.assign({}, proxy.$store.state, JSON.parse(sessionStorage.getItem("dataStore"))));
    }

    window.addEventListener("beforeunload", () => {
      sessionStorage.setItem("dataStore", JSON.stringify(proxy.$store.state));
    });
  },
}
</script>

<style lang="scss" scoped>
@import "@/assets/css/var.scss";
@import "@/assets/css/global.scss";

.el-container {
  min-height: calc(100% - 60px);
}
.el-header {
  padding: 0;
}
.el-main {
  padding-left: 0;
  padding-right: 0;
}
</style>
