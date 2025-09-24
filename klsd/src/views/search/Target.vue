<template>
  <div class="browse-container">
    <!-- 背景层 -->
    <div class="background-layer"></div>
    
    <!-- 主内容区 -->
    <div class="main-content">
      <!-- 标题和搜索区 -->
      <div class="top-section">
        <div class="header-section">
          <h1>Search Results</h1>
        </div>
        <div class="search-section">
          <el-radio-group v-model="selectedOption" class="radio-group">
            <el-radio @click="handleTarget" id="target" label="target" value="target">Target</el-radio>
            <el-radio @click="handleCompound" id="compound" label="compound" value="compound">Compound</el-radio>
          </el-radio-group>
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
              <label>Target1：</label>
              <el-input 
                v-model="input_target1" 
                placeholder="Enter target name"
                @input="handleInput"
              ></el-input>
            </div>

            <div class="filter-group">
              <label>Target2：</label>
              <el-input 
                v-model="input_target2" 
                placeholder="Enter target name"
                @input="handleInput"
              ></el-input>
            </div>

            <div class="filter-group range-filter">
              <label>Diff Range：</label>
              <div class="range-inputs">
                <el-input v-model="input_diff1" placeholder="Min"></el-input>
                <span>to</span>
                <el-input v-model="input_diff2" placeholder="Max"></el-input>
              </div>
            </div>
          </div>

          <!-- 按钮组 -->
          <div class="filter-buttons">
            <el-button @click="clearFilters" class="clear-btn">Clear</el-button>
            <el-button 
              @click="getTargetList(input_target1,input_target2,input_diff1,input_diff2)"
              class="filter-btn"
            >
               Get Target List
            </el-button>
            <el-button @click="exportToExcel" class="export-btn">
              Export to Excel
            </el-button>
          </div>
        </div>
      </div>

      <!-- 表格区 -->
      <div class="results-section">
        <div class="results-header">
          <h2>Target Results</h2>
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
            <el-table-column prop="moleculeChemblId" label="Molecule_ChEMBL_ID" align="center"></el-table-column>
            <el-table-column prop="targetname1" label="Target1" align="center"></el-table-column>
            <el-table-column prop="targetname2" label="Target2" align="center"></el-table-column>
            <el-table-column prop="diff" label="Diff" align="center"></el-table-column>
            <el-table-column prop="pact1" label="pAct1" align="center"></el-table-column>
            <el-table-column prop="pact2" label="pAct2" align="center"></el-table-column>
            <el-table-column prop="documentChemblId1" label="Ref_Target1" align="center"></el-table-column>
            <el-table-column prop="documentChemblId2" label="Ref_Target2" align="center"></el-table-column>
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
import * as XLSX from 'xlsx';
import { saveAs } from 'file-saver';

export default defineComponent({
  components: {
  },

  setup() {
    const { routerManager, changeIndex } = mixin();
    const pageSize = ref(50); // 页数
    const currentPage = ref(1); // 当前页
    const tableData = ref([]); // 激酶
    // 使用 ref 创建响应式变量
    const input_target1 = ref('');
    const input_target2 = ref('');
    const input_diff1 = ref('');
    const input_diff2 = ref('');
    const selectedOption = ref('target');
    const data = computed(() => tableData.value.slice((currentPage.value - 1) * pageSize.value, currentPage.value * pageSize.value));
    
    const clearFilters = () => {
      window.location.reload();
    };

    async function getTargetList(input_target1,input_target2,input_diff1,input_diff2) {
      let min;
      let max;
      let target;
      if(input_diff1 === null && input_diff2 === null) { 
          target = 'Name1='+input_target1+'&&Name2='+input_target2;
      }else if(input_diff1 != null && input_diff2 === null){ 
          min = '&&diff1='+input_diff1;
          target = 'Name1='+input_target1+'&&Name2='+input_target2+min;
      }else if(input_diff1 === null && input_diff2 != null){ 
          max = '&&diff2='+input_diff2;
          target = 'Name1='+input_target1+'&&Name2='+input_target2+max;
      }else{
          min = '&&diff1='+input_diff1;
          max = '&&diff2='+input_diff2;
          target = 'Name1='+input_target1+'&&Name2='+input_target2+min+max;
      }
      tableData.value = ((await HttpManager.getTargetList(target)) as ResponseBody).data;
      currentPage.value = 1;
    }

    try {
      getTargetList(input_target1,input_target2,input_diff1,input_diff2);
    } catch (error) {
      console.error(error);
    }

    // 获取target
    async function handleChangeView(item) {
      tableData.value = [];
      await getTargetList(input_target1,input_target2,input_diff1,input_diff2);
    }

    // 获取当前页
    function handleCurrentChange(val) {
      currentPage.value = val;
    }
  
    function handleTarget() {
      routerManager(RouterName.Target, { path: RouterName.Target});
    }
  
    function handleCompound() {
      routerManager(RouterName.Compound, { path: RouterName.Compound});
    }
  
    return {
      handleTarget,
      handleCompound,
      input_diff1,
      input_diff2,
      selectedOption,
      pageSize,
      currentPage,
      getTargetList,
      handleCurrentChange,
      handleChangeView,
      data,
      tableData,
      input_target1,
      input_target2,
      clearFilters,
    };
  },

  methods: {
    handleInput() {
      // 去除两端空格并判断是否为空
      this.input_target1 = this.input_target1 ? this.input_target1.trim() : null;
      this.input_target2 = this.input_target2 ? this.input_target2.trim() : null;
      this.input_diff1 = this.input_diff1 ? this.input_diff1.trim() : null;
      this.input_diff2 = this.input_diff2 ? this.input_diff2.trim() : null;
    },
    exportToExcel() {
      const data = this.data.map(row => ({
        'Molecule_ChEMBL_ID': row.moleculeChemblId,
        'Target1': row.targetname1,
        'Target2': row.targetname2,
        'Diff': row.diff,
        'pAct1': row.pact1,
        'pAct2': row.pact2,
        'Ref_Target1': row.documentChemblId1,
        'Ref_Target2': row.documentChemblId2,
      }));
      const worksheet = XLSX.utils.json_to_sheet(data);
      const workbook = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(workbook, worksheet, "Report");
      XLSX.writeFile(workbook, "report.xlsx");
    }
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

/* 单选按钮组样式 */
/* 单选按钮组样式 */
.radio-group {
  margin-left: auto;
  padding: 10px 0;
  
  ::v-deep .el-radio {
    margin-right: 20px;
    
    .el-radio__input {
      &.is-checked {
        .el-radio__inner {
          background: $primary;
          border-color: $primary;
        }
      }
    }
    
    .el-radio__label {
      font-size: 1.2rem;
      color: $text-dark;
    }
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
        
        &.export-btn {
          background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
          color: white;
          border: none;
          
          &:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(46, 125, 50, 0.4);
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