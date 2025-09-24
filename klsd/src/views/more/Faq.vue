<template>
  <div class="instruction-container">
    <h1 class="title">KLSD Operating Instructions</h1>
    <p class="introduction">
      Welcome to the Kinase Target Data Analysis Tool. This tool is designed to assist researchers in predicting and analyzing kinase-related data.
    </p>

    <div class="content-section">
      <h2 class="heading">Description of the main functions of the website</h2>
      <div class="instruction-step">
        <p style="margin-left: 20px;font-size: small">* Home: Provides an overview of the entire website and navigation entries.</p>
        <p style="margin-left: 20px;font-size: small">* Family: Allows users to search and browse according to different protein family classifications.</p>
        <p style="margin-left: 20px;font-size: small">* Drugs: Users can view drug structural information and perform related queries.</p>
        <p style="margin-left: 20px;font-size: small">* Molecule: Displays molecular structure data and offers search and filtering capabilities.</p>
        <p style="margin-left: 20px;font-size: small">* Prediction: Predicts the molecular activity values of the JAK family, and provides selectivity results and comparison of various model predictions.</p>
        <p style="margin-left: 20px;font-size: small">* More: Provides additional information, which may include news updates, research progress, etc.</p>
      </div>
      <h2 class="heading">Functional description of target difference analysis highlights</h2>
      <div class="instruction-step">
        <p style="margin-left: 20px;font-size: small">* 1. Select Kinase Family: Choose the kinase family you are interested in from the dropdown menu. A kinase family is a group of related proteins involved in specific biological processes. Your selection of a kinase family will be used for subsequent calculations.</p>
        <p style="margin-left: 20px;font-size: small">* 2. Choose diff Interval Size: Select the desired diff interval size by entering it in the respective field or dragging the slider. The diff interval size is a parameter used for calculating the pact value and affects the sensitivity and specificity of the results.</p>
        <p style="margin-left: 20px;font-size: small">* 3. Calculate pact Value: Click the "Calculate PACT Value" button, and the system will automatically compute and display the corresponding pact value based on your selected kinase family and diff interval size. The pact value is an indicator of protein interaction strength.</p>
        <p style="margin-left: 20px;font-size: small">* 4. Filter and Display: The system will filter and display data in the results table based on your chosen diff interval. The results table includes pact values, diff intervals, and corresponding reference ref_id.</p>
        <p style="margin-left: 20px;font-size: small">* 5. Predict SMILES Molecular Formula: Click the "Predict" button, and the system will generate the corresponding molecular structure diagram based on the pAct value and other data of the molecular formula, which will be displayed in the result details. The result table includes the activity value of the molecular formula, comparative activity value, and activity labels predicted by various models.</p>
        <p style="margin-left: 20px;font-size: small">* 6. Export Results: You can export the filtered results as a file for further analysis or preservation.</p>
        <div>
          <img src="@/assets/images/more/help11.png" alt="My Image" style="width: 100%;display: flex;justify-content: center; /* 水平居中 */"/>
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
    //  KlsdBackground,
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
  h4{
    color: #000;
    padding-left: 15px;
    padding-top: 5px;
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

.instruction-container {
  width: 100%;
  max-width: 800px;
  margin: auto;
  padding: 20px;
  font-family: 'Arial', sans-serif;
}

.title {
  text-align: center;
  font-size: 24px;
  margin-bottom: 16px;
}

.introduction {
  margin-bottom: 24px;
}

.content-section {
  padding: 16px;
}

.heading {
  font-size: 20px;
  margin-bottom: 12px;
}

.instruction-step {
  margin-bottom: 12px;
}

.instruction-step strong {
  font-size: 16px;
  margin-left: 10px;
}

p {
  text-align: justify; /* 文本对齐 */
  line-height: 1.6; /* 行间距 */
}
</style>
  