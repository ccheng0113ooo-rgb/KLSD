<template>
    <div class="news">
      <klsd-background></klsd-background>
      <div class="center-box">
        <div class="fixed-box">
        <!-- 此处添加你的内容 -->
        <div class="title">
          <h3>More > News</h3>
        </div>
        </div>
      </div>
    </div>
  </template>
    
    <script lang="ts">
    import { defineComponent } from "vue";
    import KlsdBackground from "@/components/layouts/KlsdBackground.vue";
    
    export default defineComponent({
      components: {
       KlsdBackground,
     },
      data() {
        return {
  
        };
      },
    });
    </script>
  
  <style lang="scss" scoped>
  @import "@/assets/css/var.scss";
  @import "@/assets/css/global.scss";
  
  .center-box {
    display: flex;
    justify-content: center;
    padding-top: 40px;
    height: 500px; /* 使用视口高度来居中 */
  }
  
  .fixed-box {
    width: 90%;
    background-color: #ffffff; /* 设置背景颜色 */
    overflow: hidden; /* 隐藏溢出部分 */
  }
  
  .fixed-box .scroll-content {
    width: 100%;
    height: 100%;
    overflow: auto; /* 添加滚动条 */
  }
  
  .title{
    padding: 10px 0;
    background: #fff;
    h3{
      color: #000;
      padding-left: 15px;
      font-weight: bold;
    }
  }
  </style>
    