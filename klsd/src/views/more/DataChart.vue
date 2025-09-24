<template>
<div class="image-container">
    <img
      class="centered-image"
      :src="imageSrc"
      :style="{ transform: `scale(${scale})` }"
      alt="Image"
    >
    <button @click="zoomIn">Larger</button>
    <button @click="zoomOut">Smaller</button>
  </div>
  </template>
    
    <script lang="ts">
    import { defineComponent, ref } from "vue";
    
    export default defineComponent({
      data() {
        return {
          scale: 1, // 初始化缩放级别
          imageSrc: require('@/assets/images/more/figure_6.png'), // 图片路径
        };
      },
      methods: {
        zoomIn() {
          this.scale += 0.1; // 按比例增加缩放级别
        },
        zoomOut() {
          if (this.scale > 0.1) { // 确保缩放级别不会低于0.1
            this.scale -= 0.1; // 按比例减少缩放级别
          }
        },
      },
    });
    </script>
  
  <style lang="scss" scoped>
  @import "@/assets/css/var.scss";
  @import "@/assets/css/global.scss";
@import "@/assets/css/table.scss";
  
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

  .cover {
    display: flex;
  justify-content: center; /* 水平居中 */
  align-items: center; /* 垂直居中 */
  height: 100%; /* 设置容器高度为视口高度，可以根据需要进行调整 */
}
.centered-image {
  max-width: 80%; /* 图片最大宽度为容器宽度 */
  max-height: 80%; /* 图片最大高度为容器高度 */
  width: 80%; /* 自动适应宽度 */
  height: 80%; /* 自动适应高度 */
}

.image-container {
  display: flex;
  justify-content: center;
  align-items: center;
  overflow: hidden;
}

.centered-image {
  transition: transform 0.2s; // 平滑变换效果
  max-width: 100%;
  max-height: 100%;
}
button {
  background-color:#D33B60;
  color: white;
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 0.25rem;
  cursor: pointer;
  font-size: 1rem;
  margin: 10px;
}
  </style>
    