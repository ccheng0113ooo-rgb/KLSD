<template>
  <div class="browse-container">
    <!-- 背景层 -->
    <div class="background-layer"></div>
    
    <!-- 主内容区 -->
    <div class="main-content">
      <!-- 标题和搜索区 -->
      <div class="top-section">
        <div class="header-section">
          <h1>Browse Molecule Structures</h1>
        </div>
        <div class="search-section">
          <div class="search-box">
            <el-input
              v-model="input_search"
              placeholder="Please enter the keywords you want to query"
              @keydown.enter="getChemblKinaseOfLikeTargetName(input_search)"
              :clearable="true"
            >
              <template #prefix>
                <i class="el-icon-search"></i>
              </template>
            </el-input>
            <div class="example-text">Examples: JAK1、JAK2、JAK3</div>
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
              <label>Family：</label>
              <el-select
                v-model="input_family"
                @change="handleChange"
                placeholder="Select family"
              >
                <el-option
                  v-for="subfamily in subfamilies"
                  :key="subfamily.id"
                  :label="subfamily.subfamilyname"
                  :value="subfamily.subfamilyname"
                ></el-option>
              </el-select>
            </div>

            <div class="filter-group range-filter">
              <label>pChEMBL Value Range：</label>
              <div class="range-inputs">
                <el-input v-model="input_pchembl1" placeholder="Min"></el-input>
                <span>to</span>
                <el-input v-model="input_pchembl2" placeholder="Max"></el-input>
              </div>
            </div>
          </div>

          <!-- 按钮组 -->
          <div class="filter-buttons">
            <el-button @click="clearFilters" class="clear-btn">Clear</el-button>
            <el-button 
              @click="getFilteredData(input_pchembl1, input_pchembl2)"
              class="filter-btn"
            >
              Get Molecule List
            </el-button>
          </div>
        </div>
      </div>

      <!-- 表格区 -->
      <div class="results-section">
        <div class="results-header">
          <h2>Molecule Results</h2>
        </div>
        
        <div class="results-content">
          <div class="pagination-controls">
            <span>Records per page：</span>
            <el-pagination
              v-model:current-page="currentPage"
              v-model:page-size="pageSize"
              :page-sizes="[1000, 2000, 5000, 10000]"
              layout="sizes, prev, pager, next"
              :total="tableData.length"
              @size-change="handleCurrentChange"
              @current-change="handleCurrentChange"
            />
            <el-button type="text" icon="el-icon-download"></el-button>
          </div>

          <!-- 加载状态 -->
          <div v-if="loading" class="loading-indicator">Loading...</div>
          
          <el-table 
            v-if="!loading"
            :data="data" 
            highlight-current-row 
            border 
            stripe
            style="width: 100%" 
            height="calc(100vh - 450px)"
          >
            <el-table-column fixed prop="moleculechemblid" label="Molecule ChEMBL ID" width="150" align="center"></el-table-column>
            <el-table-column prop="compoundkey" label="Compound Key" align="center"></el-table-column>
            <el-table-column prop="smiles" label="Smiles" width="150" align="center"></el-table-column>
            <el-table-column prop="standardtype" label="Standard Type" align="center"></el-table-column>
            <el-table-column prop="standardrelation" label="Standard Relation" align="center"></el-table-column>
            <el-table-column prop="standardvalue" label="Standard Value" align="center"></el-table-column>
            <el-table-column prop="standardunits" label="Standard Units" align="center"></el-table-column>
            <el-table-column prop="pchemblvalue" label="pChEMBL Value" align="center"></el-table-column>
            <el-table-column prop="assaychemblid" label="Assay ChEMBL ID" align="center"></el-table-column>
            <el-table-column prop="assaydescription" label="Assay Description" width="150" align="center"></el-table-column>
            <el-table-column prop="baolabel" label="BAO Label" align="center"></el-table-column>
            <el-table-column prop="assayorganism" label="Assay Organism" align="center"></el-table-column>
            <el-table-column prop="targetchemblid" label="Target ChEMBL ID" align="center"></el-table-column>
            <el-table-column prop="targetname" label="Target Name" width="150" align="center"></el-table-column>
            <el-table-column prop="targetorganism" label="Target Organism" align="center"></el-table-column>
            <el-table-column prop="targettype" label="Target Type" align="center"></el-table-column>
            <el-table-column prop="documentchemblid" label="Document ChEMBL ID" align="center"></el-table-column>
            <el-table-column prop="sourcedescription" label="Source Description" align="center"></el-table-column>
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
    const pageSize = ref(1000);
    const currentPage = ref(1);
    const tableData = ref([]);
    const input_search = ref('');
    const input_family = ref('');
    const input_pchembl1 = ref('');
    const input_pchembl2 = ref('');
    const groups = ref([]);
    const subfamilies = ref([]);
    const target = ref(null);
    const loading = ref(false);
    const data = computed(() => tableData.value.slice((currentPage.value - 1) * pageSize.value, currentPage.value * pageSize.value));
    
    const uniqueGroups = computed(() => {
      const groupSet = new Set(groups.value.map(item => item.groupname));
      return Array.from(groupSet).map(groupname => ({
        id: groupname,
        groupname,
      }));
    });

    const clearFilters = () => {
      window.location.reload();
    };

    async function getFilteredData(input_pchembl1, input_pchembl2) {
      let min;
      let max;
      let target1;
      let target2;
      target1 = 'Name1='+target.value;
      target2 = 'Name1='+target.value;
      if(input_pchembl1 === null){
        min = null;
      }else{
        min = '&&pchemblvalue1='+ input_pchembl1;
      }
      if(input_pchembl2 === null){
        max = null;
      }else{
        max = '&&pchemblvalue2='+input_pchembl2;
      }
      target1 = target1+min+max;
      if(min === '&&pchemblvalue1=' && max === '&&pchemblvalue2='){
        input_search.value = null;
        tableData.value = ((await HttpManager.getChemblKinaseOfLikePChemblValue(target2)) as ResponseBody).data;
        currentPage.value = 1;
        target2 = null;
      }
      if(min === '&&pchemblvalue1=' && max != '&&pchemblvalue2='){
        input_search.value = null;
        tableData.value = ((await HttpManager.getChemblKinaseOfLikePChemblValue(target2+max)) as ResponseBody).data;
        currentPage.value = 1;
        target2 = null;
      }
      if(min != '&&pchemblvalue1=' && max === '&&pchemblvalue2='){
        input_search.value = null;
        tableData.value = ((await HttpManager.getChemblKinaseOfLikePChemblValue(target2+min)) as ResponseBody).data;
        currentPage.value = 1;
        target2 = null;
      }
    }

    async function getChemblKinase() {
      try {
        loading.value = true;
        tableData.value = ((await HttpManager.getChemblKinase()) as ResponseBody).data;
        currentPage.value = 1;
      } catch (error) {
        console.error(error);
      } finally {
        loading.value = false;
      }
    }

    async function getChemblKinaseOfLikeTargetName(input_search) {
      try {
        loading.value = true;
        input_family.value = null;
        tableData.value = ((await HttpManager.getChemblKinaseOfLikeTargetName(input_search)) as ResponseBody).data;
        currentPage.value = 1;
      } catch (error) {
        console.error(error);
      } finally {
        loading.value = false;
      }
    }

    async function getBrowseList() {
      const response = (await HttpManager.getBrowseList()) as ResponseBody;
      groups.value = response.data.map(item => ({ id: item.id, groupname: item.groupname }));
      subfamilies.value = response.data.map(item => ({ id: item.id, subfamilyname: item.subfamilyname }));
    }

    try {
      getBrowseList();
      getChemblKinase();
    } catch (error) {
      console.error(error);
    }

    async function handleChangeView(item) {
      tableData.value = [];
      try {
        if (input_search.value ===null) {
          await getChemblKinase();
        } else {
          await getChemblKinaseOfLikeTargetName(input_search) 
        }
      } catch (error) {
        console.error(error);
      }
    }

    function handleCurrentChange(val) {
      currentPage.value = val;
    }
    
    const handleChange = (event) => {
      target.value = input_family.value;
      console.log('选中的子家族是：', target.value);
    };

    return {
      pageSize,
      loading,
      currentPage,
      tableData,
      data,
      target,
      handleChangeView,
      handleCurrentChange,
      handleChange,
      input_search,
      input_family,
      input_pchembl1,
      input_pchembl2,
      uniqueGroups,
      subfamilies,
      getChemblKinaseOfLikeTargetName,
      getFilteredData,
      clearFilters,
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
  box-shadow: $shadow-base;
  transition: all 0.3s;
  
  &:hover {
    box-shadow: $shadow-hover;
  }
  
  :deep(.el-input__wrapper) {
    border-radius: 10px;
    border: 2px solid #e0e0e0;
    padding: 15px 20px 15px 45px;
    height: auto;
    font-size: 1.2rem;
    
    &:hover, &.is-focus {
      border-color: $primary;
      box-shadow: 0 0 0 3px rgba($primary, 0.2);
    }
  }
  
  :deep(.el-input__inner) {
    font-size: 1.2rem;
    color: $text-dark;
    height: auto;
    padding: 0;
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

        :deep(.el-select) {
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
        
        &.range-filter {
          .range-inputs {
            display: flex;
            align-items: center;
            gap: 10px;
            
            :deep(.el-input) {
              flex: 1;
              
              .el-input__wrapper {
                height: 50px;
              }
            }
            
            span {
              color: $text-medium;
              font-size: 1rem;
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
        
        &.clear-btn {
          background: white;
          border: 2px solid #e0e0e0;
          color: $text-dark;
          
          &:hover {
            border-color: $primary;
            color: $primary;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba($primary, 0.2);
          }
        }
        
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

  /* 加载指示器 */
  .loading-indicator {
    position: relative;
    text-align: center;
    padding: 20px;
    font-size: 1.2rem;
    color: $primary;
    background: rgba(255, 255, 255, 0.8);
    border-radius: 8px;
    margin-bottom: 20px;
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