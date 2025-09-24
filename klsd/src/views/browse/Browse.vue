<template>
  <div class="browse-container">
    <!-- 背景层 -->
    <div class="background-layer"></div>
    
    <!-- 主内容区 -->
    <div class="main-content">
      <!-- 标题和搜索区 -->
      <div class="top-section">
        <div class="header-section">
          <h1>Browse Kinase Family</h1>
        </div>
        <div class="search-section">
          <div class="search-box">
            <el-input
              v-model="input_search"
              placeholder="Please enter the keywords you want to query"
              @keydown.enter="getBrowseListOfSubFamily(input_search)"
              :clearable="true"
            >
              <template #prefix>
                <i class="el-icon-search"></i>
              </template>
            </el-input>
            <div class="example-text">Examples: TK、JAK</div>
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
              <label>Group：</label>
              <el-select 
                v-model="input_group"
                @change="getBrowseListOfGroup(input_group)"
                placeholder="Select group"
              >
                <el-option label="All" value="allgroup"></el-option>
                <el-option 
                  v-for="option in groupValues" 
                  :key="option" 
                  :label="option" 
                  :value="option"
                ></el-option>
              </el-select>
            </div>

            <div class="filter-group">
              <label>Family：</label>
              <el-select 
                v-model="input_family"
                @change="getBrowseListOfSubFamily1(input_family)"
                placeholder="Select family"
              >
                <el-option label="All" value="allsubfamily"></el-option>
                <el-option 
                  v-for="option in subfamilyValues" 
                  :key="option" 
                  :label="option" 
                  :value="option"
                ></el-option>
              </el-select>
            </div>

            <div class="filter-group range-filter">
              <label>Number Range：</label>
              <div class="range-inputs">
                <el-input v-model="input_number1" placeholder="Min" @input="handleInput"></el-input>
                <span>to</span>
                <el-input v-model="input_number2" placeholder="Max" @input="handleInput"></el-input>
              </div>
            </div>

            <div class="filter-group range-filter">
              <label>Active Range：</label>
              <div class="range-inputs">
                <el-input v-model="input_act1" placeholder="Min" @input="handleInput"></el-input>
                <span>to</span>
                <el-input v-model="input_act2" placeholder="Max" @input="handleInput"></el-input>
              </div>
            </div>
          </div>

          <!-- 按钮组 -->
          <div class="filter-buttons">
            <el-button @click="clearFilters" class="clear-btn">Clear</el-button>
            <el-button 
              @click="getFilteredData(input_number1,input_number2,input_act1,input_act2)"
              class="filter-btn"
            >
              Get Group List
            </el-button>
          </div>
        </div>
      </div>

      <!-- 表格区 -->
      <div class="results-section">
        <div class="results-header">
          <h2>Kinase Family Results</h2>
        </div>
        
        <div class="results-content">
          <div class="pagination-controls">
            <span>Records per page：</span>
            <el-pagination
              v-model:current-page="currentPage"
              v-model:page-size="pageSize"
              :page-sizes="[10, 20, 30, 40]"
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
            border
          >
            <el-table-column prop="groupname" label="Group" align="center"></el-table-column>
            <el-table-column prop="subfamilyname" label="SubFamily" align="center"></el-table-column>
            <el-table-column prop="number" label="Number" align="center"></el-table-column>
            <el-table-column prop="active" label="Active" align="center"></el-table-column>
            <el-table-column prop="inactive" label="Inactive" align="center"></el-table-column>
          </el-table>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { ref , onMounted, nextTick, defineComponent , computed} from "vue";
import mixin from "@/mixins/mixin";
import { RouterName, NavName } from "@/enums";
import { HttpManager } from "@/api";
import "@/assets/css/table.scss";

export default defineComponent({
  components: {
  },

  setup() {
    const { routerManager, changeIndex } = mixin();
    const pageSize = ref(20);
    const currentPage = ref(1);
    const tableData = ref([]);
    const input_search = ref('');
    const input_group = ref('');
    const input_family = ref('');
    const input_number1 = ref(null);
    const input_number2 = ref(null);
    const input_act1 = ref(null);
    const input_act2 = ref(null);
    const data = computed(() => tableData.value.slice((currentPage.value - 1) * pageSize.value, currentPage.value * pageSize.value));
    
    const clearFilters = () => {
      window.location.reload();
    };

    async function getBrowseList() {
      tableData.value = ((await HttpManager.getBrowseList()) as ResponseBody).data;
      currentPage.value = 1;
    }

    async function getBrowseListOfSubFamily(input_search) {
      tableData.value = ((await HttpManager.getBrowseListOfSubFamily(input_search)) as ResponseBody).data;
      currentPage.value = 1;
      nextTick(() => {
        input_search = '';
      });
    }

    async function getBrowseListOfSubFamily1(input_family) {
      if (input_family === 'allsubfamily') {
        await getBrowseList();
      } else {
        tableData.value = ((await HttpManager.getBrowseListOfSubFamily(input_family)) as ResponseBody).data;
        currentPage.value = 1;
      }
    }

    async function getBrowseListOfGroup(input_group) {
      if (input_group === 'allgroup') {
        await getBrowseList();
      } else {
        tableData.value = ((await HttpManager.getBrowseListOfGroup(input_group)) as ResponseBody).data;
        currentPage.value = 1;
      }
    }

    async function getFilteredData(input_number1,input_number2,input_act1,input_act2) {
      if(input_number1 === null && input_number2 === null && input_act1 === null && input_act2 === null){
        await getBrowseList();
      }else if(input_act1 === null && input_act2 === null){
        let num1=null;
        let num2=null;
        let newString=null;
        if(input_number1 === null) { 
          num1='number1='+null; 
        }else { 
          num1 = 'number1=' + input_number1;
        }
        if(input_number2 === null){
          newString = num1;
        }else{
          num2 = '&&number2=' + input_number2;
          newString = num1 + num2;
        }
        tableData.value = ((await HttpManager.getNumber(newString)) as ResponseBody).data;
        currentPage.value = 1;
      }else if(input_number1 === null && input_number2 === null){
        let ac1=null;
        let ac2=null;
        let newString=null;
        if(input_act1 === null) { 
          ac1='active1='+null; 
        }else { 
          ac1 = 'active1=' + input_act1;
        }
        if(input_act2 === null){
            newString = ac1;
          }else{
            ac2 = '&&active2=' + input_act2;
            newString = ac1+ac2;
        }
        tableData.value = ((await HttpManager.getActive(newString)) as ResponseBody).data;
        currentPage.value = 1;
      }else{
        let num1=null;
        let num2=null;
        let ac1=null;
        let ac2=null;
        let newString=null;
        if(input_number1 === null) { 
          num1='number1=null'; 
          num2 = '&&number2=' + input_number2;
        }else { 
          num1 = 'number1=' + input_number1;
          if(input_number2 === null) { 
            num2=''; 
          }else { 
            num2 = '&&number2=' + input_number2;
          }
        }
        newString = num1+num2;
        if(input_act1 === null) { 
          ac1 = '&&active1=null';
          ac2 = '&&active2=' + input_act2; 
        }else { 
          ac1 = '&&active1=' + input_act1; 
          if(input_act2 === null) { 
            ac2=''; 
          }else { 
            ac2 = '&&active2=' + input_act2;
          }
        }
        newString = newString + ac1 + ac2;
        tableData.value = ((await HttpManager.getNumberandActive(newString)) as ResponseBody).data;
        currentPage.value = 1;
      }
    }

    try {
      getBrowseList();
    } catch (error) {
      console.error(error);
    }

    function handleCurrentChange(val) {
      currentPage.value = val;
    }

    return {
      tableData,
      input_search,
      input_group,
      input_family,
      input_number1,
      input_number2,
      input_act1,
      input_act2,
      pageSize,
      currentPage,
      data,
      handleCurrentChange,
      getBrowseListOfSubFamily,
      getBrowseListOfSubFamily1,
      getBrowseListOfGroup,
      getFilteredData,
      clearFilters,
    };
  },

  computed: {
    groupValues() {
      const uniqueValues = new Set();
      for (const item of this.tableData) {
        uniqueValues.add(item.groupname);
      }
      return Array.from(uniqueValues);
    },
    subfamilyValues() {
      const uniqueValues = new Set();
      for (const item of this.tableData) {
        uniqueValues.add(item.subfamilyname);
      }
      return Array.from(uniqueValues);
    },
  },
  methods: {
    handleInput() {
      this.input_number1 = this.input_number1 ? this.input_number1.trim() : null;
      this.input_number2 = this.input_number2 ? this.input_number2.trim() : null;
      this.input_act1 = this.input_act1 ? this.input_act1.trim() : null;
      this.input_act2 = this.input_act2 ? this.input_act2.trim() : null;
    }
  },
});
</script>

<style lang="scss" scoped>
/* 变量定义 - 与其他页面完全同步 */
$primary: #1976D2;
$primary-light: #42A5F5;
$secondary: #764ba2;
$gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
$gradient-bg: linear-gradient(135deg, rgb(255, 255, 255) 0%, rgb(206, 222, 241) 100%);
$text-light: #ffffff;
$text-dark: #2c3e50;
$text-medium: #666;
$danger: #F44336;
$shadow-base: 0 5px 20px rgba(0, 0, 0, 0.1);
$shadow-hover: 0 15px 40px rgba(0, 0, 0, 0.15);

/* 基础容器 - 强化层级与其他页一致 */
.browse-container {
  position: relative;
  min-height: 100vh;
  padding: 20px;
  background: $gradient-bg;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  zoom: 0.8;
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

  > div {
    position: relative;
    z-index: 1;
  }
}

/* 头部区域 - 文字风格与其他页对齐 */
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

/* 搜索区域 - 输入框样式与其他页严格匹配 */
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

/* 筛选区域 - 卡片化样式与其他页卡片统一 */
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

        :deep(.el-select){
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
      }
    }
  }
}

/* 结果区域 - 与其他页结果卡片风格统一 */
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

/* 响应式设计 - 与其他页适配逻辑一致 */
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