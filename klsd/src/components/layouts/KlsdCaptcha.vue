<template>
    <div>
      <!-- 显示验证码图片，点击图片刷新验证码 -->
      <img :src="captchaImage" alt="Captcha" @click="refreshCaptcha" />
    </div>
  </template>
  
  <script>
  export default {
    data() {
      return {
        captchaImage: "" // 验证码图片的URL
      };
    },
    mounted() {
      this.refreshCaptcha(); // 组件挂载时刷新验证码
    },
    methods: {
      refreshCaptcha() {
        this.captchaImage = `/api/captcha?t=${Date.now()}`; // 刷新验证码图片的URL，添加时间戳以避免缓存
      }
    }
  };
  </script>
  
  <style scoped>
  img {
    cursor: pointer;  
  }
  </style>
  