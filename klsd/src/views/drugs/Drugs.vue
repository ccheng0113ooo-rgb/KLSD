<template>
  <div class="browse-container">
    <!-- 背景层 -->
    <div class="background-layer"></div>
    
    <!-- 主内容区 -->
    <div class="main-content">
      <!-- 标题和搜索区 -->
      <div class="top-section">
        <div class="header-section">
          <h1>Browse Drugs Structures</h1>
        </div>
        <div class="search-section">
          <div class="search-box">
            <el-input
              v-model="input_search"
              placeholder="Please enter the keywords you want to query"
              @keydown.enter="getDrugsSearchList1(input_search)"
              :clearable="true"
            >
              <template #prefix>
                <i class="el-icon-search"></i>
              </template>
            </el-input>
            <div class="example-text">Examples: GEFITINIB DASATINIB BOSUTINIB REGORAFENIB SECRETIN</div>
          </div>
        </div>
      </div>

      <!-- 筛选区 -->
      <div class="filter-section">
        <div class="filter-header">
          <h2>Filters</h2>
        </div>
        <div class="filter-content">
          <!-- 筛选框组 -->
          <div class="filter-controls">
            <div class="filter-group">
              <label>Name：</label>
              <el-input 
                v-model="input_name"
                placeholder="Enter drug name"
                @keydown.enter="getDrugsSearchList(input_name)"
                :clearable="true"
              ></el-input>
            </div>
          </div>

          <!-- 按钮组 -->
          <div class="filter-buttons">
            <el-button @click="getDrugsSearchList(input_name)" class="filter-btn">
              Get Drugs List
            </el-button>
          </div>
        </div>
      </div>

      <!-- 表格区 -->
      <div class="results-section">
        <div class="results-header">
          <h2>Drugs Results</h2>
        </div>
        
        <div class="results-content">
          <div class="pagination-controls">
            <span>Records per page：</span>
            <el-pagination
              v-model:current-page="currentPage"
              v-model:page-size="pageSize"
              :page-sizes="[50, 100, 200, 500]"
              layout="sizes, prev, pager, next"
              :total="tableData.length"
              @size-change="handleCurrentChange"
              @current-change="handleCurrentChange"
            />
            <el-button type="text" icon="el-icon-download"></el-button>
          </div>

          <el-table 
            :data="data" 
            style="width: 100%" 
            height="calc(100vh - 450px)"
            stripe
          >
            <el-table-column fixed prop="moleculeChemblId" label="Molecule ChEMBL ID" align="center"></el-table-column>
            <el-table-column prop="drugName" label="Drug_Name" align="center"></el-table-column>
            <el-table-column prop="name" label="Name" align="center"></el-table-column>
            <el-table-column prop="standardType" label="Standard Type" align="center"></el-table-column>
            <el-table-column prop="standardRelation" label="Standard Relation" width="200" align="center"></el-table-column>
            <el-table-column prop="standardValue" label="Standard Value" width="200" align="center"></el-table-column>
            <el-table-column prop="standardUnits" label="Standard Units" width="200" align="center"></el-table-column>
            <el-table-column prop="documentChemblId" label="Document ChEMBL ID" width="200" align="center"></el-table-column>
          </el-table>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { ref , onMounted, defineComponent , computed} from "vue";
import mixin from "@/mixins/mixin";
import { RouterName, NavName } from "@/enums";
import { HttpManager } from "@/api";
import "@/assets/css/table.scss";

export default defineComponent({
  components: {
  },

  setup() {
    const { routerManager, changeIndex } = mixin();
    const pageSize = ref(50); // 页数
    const currentPage = ref(1); // 当前页
    const tableData = ref([]); // drugs
    const input_name = ref('');
    const input_search = ref('');
    const data = computed(() => tableData.value.slice((currentPage.value - 1) * pageSize.value, currentPage.value * pageSize.value));
    
    async function getDrugsList() {
      tableData.value = ((await HttpManager.getDrugsList()) as ResponseBody).data;
      currentPage.value = 1;
    }

    async function getDrugsSearchList(input_name) {
      tableData.value = ((await HttpManager.getDrugsListOfLikeName(input_name)) as ResponseBody).data;
      currentPage.value = 1;
      input_search.value='';
    }

    async function getDrugsSearchList1(input_search) {
      tableData.value = ((await HttpManager.getDrugsListOfLikeDrugName(input_search)) as ResponseBody).data;
      currentPage.value = 1;
      input_name.value='';
    }

    try {
      getDrugsList();
    } catch (error) {
      console.error(error);
    }

    // 获取drug
    async function handleChangeView(item) {
      tableData.value = [];
      try {
        if (input_name.value === null && input_search.value ===null) {
          await getDrugsList();
        } else if(input_search.value === null){
          await getDrugsSearchList(input_name);
        }else {
          await getDrugsSearchList1(input_search);
        }
      } catch (error) {
        console.error(error);
      }
    }
    
    // 获取当前页
    function handleCurrentChange(val) {
      currentPage.value = val;
    }

    return {
      input_name,
      input_search,
      pageSize,
      currentPage,
      tableData,
      data,
      handleChangeView,
      handleCurrentChange,
      getDrugsSearchList,
      getDrugsSearchList1,
    };
  },
});
</script>

<style lang="scss" scoped>
/* 变量定义 - 与browse页面完全同步 */
$primary: #1976D2;
$primary-light: #42A5F5;
$secondary: #764ba2;
$gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%); /* 核心渐变变量名统一 */
$gradient-bg: linear-gradient(135deg, rgb(255, 255, 255) 0%, rgb(206, 222, 241) 100%);
$text-light: #ffffff;
$text-dark: #2c3e50;
$text-medium: #666;
$danger: #F44336;
$shadow-base: 0 5px 20px rgba(0, 0, 0, 0.1); /* 基础阴影变量 */
$shadow-hover: 0 15px 40px rgba(0, 0, 0, 0.15); /* 悬停阴影变量 */

/* 基础容器 - 强化层级与browse页一致 */
.browse-container {
  position: relative;
  min-height: 100vh;
  padding: 20px;
  background: $gradient-bg;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('@/assets/images/home/homeback.png') center/cover;
    opacity: 0.08;
    z-index: 0;
    pointer-events: none;
  }

  /* 所有内容容器统一层级，覆盖背景图 */
  > div {
    position: relative;
    z-index: 1;
  }
}

/* 头部区域 - 文字风格与browse页对齐 */
.top-section {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  margin-bottom: 30px;
  align-items: center;
}

.header-section {
  flex: 1;
  
  h1 {
    font-size: 2.1rem;
    color: $primary;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin: 0;
    padding: 0;
  }
}

/* 搜索区域 - 输入框样式与browse页输入框严格匹配 */
.search-section {
  flex: 1;
  max-width: 800px;
}

.search-box {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: $shadow-base; /* 使用统一阴影变量 */
  transition: all 0.3s; /* 新增悬停过渡 */
  
  &:hover {
    box-shadow: $shadow-hover; /* 与browse页卡片悬停阴影一致 */
  }
  
  :deep(.el-input__wrapper) {
    border-radius: 10px; /* 与browse页输入框圆角一致 */
    border: 2px solid #e0e0e0;
    padding: 15px 20px 15px 45px; /* 增加内边距，与browse页输入框高度匹配 */
    height: auto;
    font-size: 1.2rem;
    
    &:hover, &.is-focus {
      border-color: $primary;
      box-shadow: 0 0 0 3px rgba($primary, 0.2); /* 与browse页输入框聚焦效果一致 */
    }
  }
  
  :deep(.el-input__inner) {
    font-size: 1.2rem;
    color: $text-dark;
    height: auto; /* 取消固定高度，由内边距撑开 */
    padding: 0; /* 清除默认内边距 */
  }
  
  :deep(.el-icon-search) {
    color: $primary;
    font-size: 1.3rem;
    margin-left: 10px;
  }
  
  .example-text {
    margin-top: 15px;
    color: $text-medium;
    font-size: 1rem;
    text-align: left;
    padding-left: 5px;
  }
}

/* 筛选区域 - 卡片化样式与browse页卡片统一 */
.filter-section {
  background: white;
  border-radius: 12px;
  margin-bottom: 50px;
  box-shadow: $shadow-base;
  overflow: hidden;
  transition: all 0.3s;
  opacity: 0.8;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: $shadow-hover;
  }

  .filter-header {
    background: $gradient-primary;
    padding: 15px 25px;
    
    h2 {
      color: white;
      font-size: 1.7rem;
      margin: 0;
      font-weight: 500;
      text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
  }

  .filter-content {
    padding: 25px;
    
    .filter-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 30px;
      margin-bottom: 25px;
      
      .filter-group {
        flex: 1;
        min-width: 250px;

        label {
          display: block;
          margin-bottom: 15px;
          font-size: 1.3rem;
          color: $text-dark;
          font-weight: 600;
        }

        :deep(.el-input) {
          width: 100%;

          .el-input__wrapper {
            height: 50px;
            padding: 0 15px;
            font-size: 1.1rem;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            
            &:hover, &.is-focus {
              border-color: $primary;
              box-shadow: 0 0 0 3px rgba($primary, 0.2);
            }
          }
        }
      }
    }

    .filter-buttons {
      display: flex;
      justify-content: flex-start;
      gap: 20px;
      
      .el-button {
        height: 50px;
        padding: 0 30px;
        font-size: 1.2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
        
        &.filter-btn {
          background: $gradient-primary;
          color: white;
          border: 1px solid #667eea;
          
          &:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba($primary, 0.4);
          }
        }
      }
    }
  }
}

/* 结果区域 - 与browse页结果卡片风格统一 */
.results-section {
  background: white;
  border-radius: 12px;
  margin-bottom: 30px;
  box-shadow: $shadow-base;
  overflow: hidden;
  transition: all 0.3s;
  opacity: 0.8;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: $shadow-hover;
  }

  .results-header {
    background: $gradient-primary;
    padding: 15px 25px;
    
    h2 {
      color: white;
      font-size: 1.7rem;
      margin: 0;
      font-weight: 500;
      text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
  }

  .results-content {
    padding: 25px;
    overflow-x: auto;
  }

  /* 表格样式精细化调整 */
  :deep(.el-table) {
    width: 100% !important;
    font-size: 1.1rem;
    border-radius: 8px;
    overflow: hidden;
    
    /* 表头样式 */
    th {
      font-weight: 600;
      color: $text-dark;
      background-color: #f8f9fa !important;
      padding: 15px 0;
      border-bottom: 1px solid #eee;
    }
    
    /* 单元格样式 */
    td {
      color: $text-dark;
      padding: 15px 0;
      border-bottom: 1px solid #f1f1f1;
    }
  }

  /* 分页控件 - 样式统一 */
  .pagination-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 25px;
    padding: 0 15px;
    
    .records-per-page {
      font-size: 1rem;
      color: $text-medium;
      display: flex;
      align-items: center;
    }
    
    :deep(.el-pagination) {
      .btn-prev, .btn-next, .number {
        font-size: 1.1rem;
        min-width: 36px;
        height: 36px;
        line-height: 36px;
        border-radius: 6px;
      }

      .active {
        background-color: $primary;
        color: white;
      }
    }
  }
}

/* 响应式设计 - 与browse页适配逻辑一致 */
@media (max-width: 768px) {
  .header-section h1 {
    font-size: 1.8rem;
    margin-bottom: 15px;
  }
  
  .filter-controls {
    flex-direction: column !important;
    gap: 20px !important;
  }
  
  .filter-buttons {
    flex-direction: column;
    gap: 15px !important;
    
    .el-button {
      width: 100%;
    }
  }

  .results-section {
    :deep(.el-table) {
      th, td {
        display: block;
        width: 100% !important;
        text-align: left !important;
        padding: 12px 15px !important;
      }
      
      tr {
        display: block;
        margin-bottom: 15px;
        border-bottom: 2px solid #eee;
      }
    }
    
    .pagination-controls {
      flex-direction: column;
      gap: 15px;
      align-items: flex-start;
    }
  }
}

@media (max-width: 992px) {
  .results-section {
    :deep(.el-table) {
      th, td {
        padding: 12px 8px !important;
      }
    }
  }
}
</style>