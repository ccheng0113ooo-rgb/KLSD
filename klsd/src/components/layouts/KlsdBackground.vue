<template>
    <div class="klsd-background">
      <div class="background"></div>
    </div>
  </template>
    
<script lang="ts">
import { defineComponent } from "vue";
    
export default defineComponent({
  data() {
    return {

    };
    },
});
</script>
  
<style lang="scss" scoped>
@import "@/assets/css/var.scss";
@import "@/assets/css/global.scss";
  
.background-container {
  position: relative;
  width: 100%;
  height: 100vh;
  overflow: hidden;
}

.content-container {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  background-color: #fff;
  padding: 20px;
}

.background {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-image: url("~@/assets/images/more/background_1.jpg");
  background-size: cover;
  z-index: -1; /* 将背景置于底层 */
  opacity: 0.3; /* 设置背景透明度 */
}

</style>