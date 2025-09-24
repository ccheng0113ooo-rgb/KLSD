<template>
  <div class="news-container">
    <div class="news-list">
      <div
        v-for="post in blogPosts"
        :key="post.title"
        class="news-item"
        @click="goToPost(post.linkUrl)"
      >
        <div class="news-content">
          <h3 class="news-title">{{ post.title }}</h3>
          <p class="news-summary">{{ post.summary }}</p>
          <div class="news-meta">
            <span class="news-date">{{ post.datetime }}</span>
            <a :href="post.linkUrl" class="news-link">Read More</a>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>


    
    <script lang="ts">
import { ref,defineComponent } from "vue";
import KlsdBackground from "@/components/layouts/KlsdBackground.vue";
import 'swiper/css/swiper.css';
import 'swiper/js/swiper.min.js';
import "@/assets/css/table.scss";
    
    export default {
      components: {
      //  KlsdBackground,
     },

     setup(){
      const blogPosts = ref([
      // ... 你的文章数据
      ]);

    // 点击文章时调用此方法
    const goToPost = (url) => {
          window.location.href = url;
    };
    return{
      goToPost,
    };
    }, 

     
      data() {
        return {
        blogPosts:[
        {
            title: 'Structural basis of a redox-dependent con- formational switch that regulates the stress kinase p38α',
            summary: 'p38α is a kinase that regulates many cellular processes. Here authors report that redox fluctuations induce conformational changes to fine-tune the p38α activity beyond its activation loop phosphorylation.',
            references: 'by Nature Communications',
            datetime:'01 Dec 2023',
            // imageUrl: require('./assets/images/news/news_01.png'),
            linkUrl: 'https://www.nature.com/articles/s41467-023-43763-5'
        },
        {
          title: 'Chemical proteomics reveals the target landscape of 1,000 kinase inhibitors',
            summary: 'Chemical proteomics profiling of 1,183 kinase inhibitors from past drug discovery projects revealed a large number of reasonably selective compounds for several kinases such as SYK and CK2.',
            references: 'by Nature Chemical Biology',
            datetime:'30 Oct 2023',
            // imageUrl: require('./assets/images/news/news_01.png'),
            linkUrl: 'https://www.nature.com/articles/s41589-023-01459-3'
        },
        {
          title: 'Global phosphoproteomics reveal the diverse roles of casein kinase 1 in plant development',
            summary: 'Protein phosphorylation is a major posttranslational modification carried out by protein kinases that constitute an integral part of complex signaling networks in eukaryotes. Casein kinase 1 (CK1) is a conserved serine/threonine ...',
            references: 'by Science China Press',
            datetime:'28 Sep 2023',
            // imageUrl: require('./assets/images/news/news_01.png'),
            linkUrl: 'https://phys.org/news/2023-09-global-phosphoproteomics-reveal-diverse-roles.html'
        }
        ],
        
      };
      },
    };
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

  .cover {
    display: flex;
  justify-content: center; /* 水平居中 */
  align-items: center; /* 垂直居中 */
  height: 100%; /* 设置容器高度为视口高度，可以根据需要进行调整 */
}
.centered-image {
  max-width: 100%; /* 图片最大宽度为容器宽度 */
  max-height: 100%; /* 图片最大高度为容器高度 */
  width: auto; /* 自动适应宽度 */
  height: auto; /* 自动适应高度 */
}

.news-container {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #333;
  background-color: #f5f5f5;
  padding: 20px;
}

.news-list {
  max-width: 800px; // Adjust the width as necessary
  margin: auto;
  display: flex;
  flex-direction: column;
}

.news-item {
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
  transition: transform 0.3s ease;

  &:hover {
    transform: translateY(-5px);
  }
}

.news-content {
  padding: 15px;
}

.news-title {
  font-size: 1.5em;
  color: #000;
  margin-bottom: 10px;
}

.news-summary {
  font-size: 1em;
  line-height: 1.6;
  color: #666;
}

.news-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 15px;
  border-top: 1px solid #eee;
  padding-top: 10px;
}

.news-date {
  font-size: 0.9em;
  color: #999;
}

.news-link {
  text-decoration: none;
  color: #007bff;
  &:hover {
    text-decoration: underline;
  }
}
  </style>
    