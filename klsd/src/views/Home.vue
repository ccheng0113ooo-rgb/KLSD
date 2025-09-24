<template>
  <!-- 主页容器 -->
  <div class="homepage-container">
    <!-- Hero区域 - 搜索功能 -->
    <section class="hero-section">
      <div class="hero-content">
        <div class="hero-header">
          <h1 class="main-title">KLSD</h1>
          <h2 class="subtitle">Kinase-Ligand Structure Database</h2>
          <p class="description">
            A comprehensive database for kinase inhibitor prediction and analysis
          </p>
        </div>
        
        <div class="search-container">
          <div class="search-box">
            <el-form @submit.prevent="handleSubmit" class="search-form">
              <el-form-item>
                <el-input 
                  v-model="smilesInput"
                  placeholder="Enter SMILES string (e.g., CN1C=NC2=C1C(=O)N(C(=O)N2C)C)"
                  clearable
                  class="search-input"
                  size="large"
                >
                  <template #prefix>
                    <i class="el-icon-search search-icon"></i>
                  </template>
                </el-input>
              </el-form-item>
              
              <div class="search-actions">
                <el-button 
                  type="primary" 
                  native-type="submit"
                  :disabled="!smilesInput"
                  class="predict-button"
                  size="large"
                >
                  <i class="el-icon-cpu"></i>
                  Predict Activity
                </el-button>
                <el-button 
                  @click="clearInput" 
                  class="clear-button"
                  size="large"
                >
                  Clear
                </el-button>
              </div>
            </el-form>
          </div>
          
          <div class="example-molecules">
            <span class="example-label">Example molecules:</span>
            <div class="molecule-tags">
              <el-tag
                v-for="(example, index) in exampleMolecules" 
                :key="index"
                class="molecule-tag"
                @click="useExample(example)"
                effect="plain"
              >
                {{ example.substring(0, 30) }}...
              </el-tag>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- 功能模块区域 -->
    <section class="features-section">
      <div class="container">
        <h2 class="section-title">Database Features</h2>
        <div class="features-grid">
          <!-- Family模块 -->
          <div class="feature-card" @click="navigateTo('/browse')">
            <div class="feature-icon family-icon">
              <i class="el-icon-collection"></i>
            </div>
            <h3 class="feature-title">Kinase Families</h3>
            <p class="feature-description">
              Explore 138 kinase families with detailed classification and structural information
            </p>
            <div class="feature-stats">
              <span class="stat-number">138</span>
              <span class="stat-label">Families</span>
            </div>
          </div>

          <!-- Search模块 -->
          <div class="feature-card" @click="navigateTo('/compound')">
            <div class="feature-icon search-icon">
              <i class="el-icon-search"></i>
            </div>
            <h3 class="feature-title">Advanced Search</h3>
            <p class="feature-description">
              Search compounds, kinases, and activities with powerful filtering options
            </p>
            <div class="feature-stats">
              <span class="stat-number">690K</span>
              <span class="stat-label">Compounds</span>
            </div>
          </div>

          <!-- Prediction模块 -->
          <div class="feature-card" @click="navigateTo('/compound-prediction')">
            <div class="feature-icon prediction-icon">
              <i class="el-icon-cpu"></i>
            </div>
            <h3 class="feature-title">Active Prediction</h3>
            <p class="feature-description">
              Predict kinase inhibitor activity using advanced machine learning models
            </p>
            <div class="feature-stats">
              <span class="stat-number">1.7M</span>
              <span class="stat-label">Activities</span>
            </div>
          </div>

          <!-- Drugs模块 -->
          <div class="feature-card" @click="navigateTo('/drugs')">
            <div class="feature-icon drugs-icon">
              <i class="el-icon-medicine"></i>
            </div>
            <h3 class="feature-title">Drug Database</h3>
            <p class="feature-description">
              Browse approved and investigational kinase inhibitor drugs
            </p>
            <div class="feature-stats">
              <span class="stat-number">4.5K</span>
              <span class="stat-label">Drugs</span>
            </div>
          </div>

          <!-- Molecule模块 -->
          <div class="feature-card" @click="navigateTo('/molecules')">
            <div class="feature-icon molecule-icon">
              <i class="el-icon-connection"></i>
            </div>
            <h3 class="feature-title">Molecular Structures</h3>
            <p class="feature-description">
              Visualize and analyze 3D molecular structures and binding sites
            </p>
            <div class="feature-stats">
              <span class="stat-number">324K</span>
              <span class="stat-label">Structures</span>
            </div>
          </div>

          <!-- About模块 -->
          <div class="feature-card" @click="navigateTo('/about')">
            <div class="feature-icon about-icon">
              <i class="el-icon-info"></i>
            </div>
            <h3 class="feature-title">About KLSD</h3>
            <p class="feature-description">
              Learn about our database, methodology, and research applications
            </p>
            <div class="feature-stats">
              <span class="stat-number">2024</span>
              <span class="stat-label">Latest</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- 数据统计区域 - 修复不一致问题 -->
    <section class="statistics-section">
      <div class="container">
        <div class="stats-grid">
          <!-- 左侧：数据库统计信息 -->
          <div class="stats-chart">
            <h3 class="chart-title">Database Statistics Overview</h3>
            <div class="chart-container">
              <canvas ref="databaseStatsChart"></canvas>
            </div>
            <div class="chart-legend">
              <div 
                v-for="(item, index) in databaseStats" 
                :key="index"
                class="legend-item"
              >
                <span 
                  class="legend-color" 
                  :style="{ backgroundColor: databaseColors[index] }"
                ></span>
                <span class="legend-label">{{ item.target }}</span>
                <span class="legend-value">{{ item.count }}</span>
              </div>
            </div>
          </div>

          <!-- 右侧：药物分子类型统计 -->
          <div class="drug-stats">
            <h3 class="chart-title">Drugs by Molecule Type</h3>
            <div class="chart-container">
              <canvas ref="drugTypeChart"></canvas>
            </div>
            <div class="drug-summary">
              <div class="summary-grid">
                <div 
                  v-for="(drug, index) in drugTypeStats" 
                  :key="index"
                  class="summary-item"
                >
                  <div class="summary-icon" :style="{ backgroundColor: drugTypeColors[index] }">
                    <i :class="drug.icon"></i>
                  </div>
                  <div class="summary-content">
                    <span class="summary-label">{{ drug.type }}</span>
                    <span class="summary-count">{{ drug.count }}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 激酶分类树状图 - 独立区域 -->
        <div class="kinome-section">
          <h2 class="section-title">Human Kinome Classification</h2>
          <div class="kinome-content">
            <div class="kinome-image-container">
              <img 
                src="../assets/images/home/human_kinase.png" 
                alt="Human Kinome Tree"
                class="kinome-image"
                @click="redirectToKinome"
              >
            </div>
            <div class="kinome-info">
              <h4 class="kinome-subtitle">Kinase Family Distribution</h4>
              <div class="family-grid">
                <div 
                  v-for="(family, index) in kinaseFamilyData" 
                  :key="family.name"
                  class="family-item"
                >
                  <span class="family-name">{{ family.name }}</span>
                  <span class="family-count">{{ family.count }}</span>
                  <div class="family-bar">
                    <div 
                      class="family-progress" 
                      :style="{ 
                        width: family.percentage + '%',
                        backgroundColor: kinaseFamilyColors[index % kinaseFamilyColors.length]
                      }"
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- 研究新闻区域 -->
    <section class="news-section">
      <div class="container">
        <div class="news-header">
          <h2 class="section-title">Latest Research</h2>
          <el-button 
            type="text" 
            class="more-button"
            @click="goToNews"
          >
            View All News
            <i class="el-icon-arrow-right"></i>
          </el-button>
        </div>
        
        <div class="news-grid">
          <div 
            v-for="(news, index) in newsList" 
            :key="index" 
            class="news-card"
            @click="redirectToNews(news.link)"
          >
            <div class="news-image">
              <img :src="news.image" :alt="news.title">
              <div class="news-overlay">
                <i class="el-icon-view"></i>
              </div>
            </div>
            <div class="news-content">
              <div class="news-meta">
                <span class="news-source">{{ news.source }}</span>
                <span class="news-date">{{ news.date }}</span>
              </div>
              <h4 class="news-title">{{ news.title }}</h4>
              <p class="news-summary">{{ news.summary }}</p>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- 关于数据库区域 -->
    <section class="about-section">
      <div class="container">
        <div class="about-content">
          <div class="about-text">
            <h2 class="about-title">What is KLSD?</h2>
            <p class="about-description">
              The Kinase-Ligand Structure Database (KLSD) is a comprehensive resource for kinase inhibitor 
              research and drug discovery. Our database integrates structural, chemical, and biological data 
              to provide researchers with powerful tools for understanding kinase-inhibitor interactions.
            </p>
            <div class="about-features">
              <div class="about-feature">
                <i class="el-icon-cpu"></i>
                <span>AI-powered prediction models</span>
              </div>
              <div class="about-feature">
                <i class="el-icon-data-analysis"></i>
                <span>Comprehensive data integration</span>
              </div>
              <div class="about-feature">
                <i class="el-icon-connection"></i>
                <span>Interactive visualization tools</span>
              </div>
            </div>
          </div>
          <div class="about-visual">
            <div class="kinase-illustration">
              <img src="../assets/images/home/homeback.png" alt="Kinase Structure">
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage } from 'element-plus';
import Chart from 'chart.js/auto';

export default defineComponent({
  name: 'HomePage',
  setup() {
    const router = useRouter();
    const smilesInput = ref('');
    const databaseStatsChart = ref<HTMLCanvasElement | null>(null);
    const drugTypeChart = ref<HTMLCanvasElement | null>(null);
    
    // 示例分子数据
    const exampleMolecules = [
      "COc1ccc(NC(=O)N2CCN3C(=O)c4ccccc4C23c2ccc(Cl)cc2)cc1",  
      "Cc1ccncc1C(=O)N1CCN(C(=O)C(=O)c2c[nH]c3ccccc23)CC1",  
      "N#C[C@H]1CCOC[C@@H]1n1cc(C(N)=O)c(Nc2ccc(C3CC3)cc2)n1"   
    ];
    
    // 数据库统计信息（左侧图表）
    const databaseStats = [
      { target: 'Activities', count: '1,686,997' },
      { target: 'Compounds', count: '690,063' },
      { target: 'Structures', count: '324,360' },
      { target: 'Active', count: '431,811' },
      { target: 'Inactive', count: '258,250' },
      { target: 'Families', count: '138' }
    ];

    // 药物分子类型统计（右侧图表）
    const drugTypeStats = [
      { type: 'Small Molecule', count: '3,245', icon: 'el-icon-connection' },
      { type: 'Protein Kinase', count: '892', icon: 'el-icon-cpu' },
      { type: 'Antibody', count: '234', icon: 'el-icon-medicine' },
      { type: 'Peptide', count: '150', icon: 'el-icon-collection' }
    ];

    // 激酶家族数据（带统计信息）
    const kinaseFamilyData = [
      { name: 'TK (Tyrosine Kinase)', count: '58', percentage: 42 },
      { name: 'TKL (Tyrosine Kinase-Like)', count: '43', percentage: 31 },
      { name: 'STE (Sterile)', count: '47', percentage: 34 },
      { name: 'CMGC', count: '61', percentage: 44 },
      { name: 'CKI (Casein Kinase I)', count: '12', percentage: 9 },
      { name: 'AGC', count: '63', percentage: 46 },
      { name: 'CAMK', count: '74', percentage: 54 },
      { name: 'Atypical', count: '40', percentage: 29 }
    ];

    // 颜色方案
    const databaseColors = [
      '#1976D2', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4'
    ];

    const drugTypeColors = [
      '#667eea', '#764ba2', '#f093fb', '#f5576c'
    ];

    const kinaseFamilyColors = [
      '#1976D2', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#795548', '#607D8B'
    ];

    // 新闻数据
    const newsList = [
      {
        title: 'Structural basis of redox-dependent conformational switch in p38α kinase',
        summary: 'New insights into p38α kinase regulation through redox-dependent conformational changes beyond phosphorylation.',
        source: 'Nature Communications',
        date: '2023-12-01',
        link: 'https://www.nature.com/articles/s41467-023-43763-5',
        image: require('../assets/images/home/news_01.png')
      },
      {
        title: 'Chemical proteomics reveals target landscape of 1,000 kinase inhibitors',
        summary: 'Comprehensive profiling of kinase inhibitor selectivity using chemical proteomics approaches.',
        source: 'Nature Chemical Biology',
        date: '2023-10-30',
        link: 'https://www.nature.com/articles/s41589-023-01459-3',
        image: require('../assets/images/home/news_03.png')
      },
      {
        title: 'Global phosphoproteomics reveal diverse roles of casein kinase 1',
        summary: 'Systematic analysis of CK1 substrates and their roles in plant development and signaling.',
        source: 'Science China Press',
        date: '2023-09-28',
        link: 'https://phys.org/news/2023-09-global-phosphoproteomics-reveal-diverse-roles.html',
        image: require('../assets/images/home/news_02.png')
      }
    ];

    // 初始化数据库统计图表
    const initDatabaseStatsChart = () => {
      if (!databaseStatsChart.value) return;

      const values = databaseStats.map(item => parseFloat(item.count.replace(/,/g, '')));
      const ctx = databaseStatsChart.value.getContext('2d');
      if (!ctx) return;

      new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: databaseStats.map(item => item.target),
          datasets: [{
            data: values,
            backgroundColor: databaseColors,
            borderWidth: 2,
            borderColor: '#fff'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false
            },
            tooltip: {
              backgroundColor: 'rgba(0, 0, 0, 0.8)',
              titleColor: '#fff',
              bodyColor: '#fff',
              borderColor: '#1976D2',
              borderWidth: 1
            }
          },
          cutout: '60%'
        }
      });
    };

    // 初始化药物类型图表
    const initDrugTypeChart = () => {
      if (!drugTypeChart.value) return;

      const values = drugTypeStats.map(item => parseInt(item.count.replace(/,/g, '')));
      const ctx = drugTypeChart.value.getContext('2d');
      if (!ctx) return;

      new Chart(ctx, {
        type: 'bar',
        data: {
          labels: drugTypeStats.map(item => item.type),
          datasets: [{
            data: values,
            backgroundColor: drugTypeColors,
            borderWidth: 0,
            borderRadius: 8
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false
            },
            tooltip: {
              backgroundColor: 'rgba(0, 0, 0, 0.8)',
              titleColor: '#fff',
              bodyColor: '#fff'
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              grid: {
                color: 'rgba(0, 0, 0, 0.1)'
              }
            },
            x: {
              grid: {
                display: false
              }
            }
          }
        }
      });
    };

    onMounted(() => {
      initDatabaseStatsChart();
      initDrugTypeChart();
    });

    // 表单提交
    const handleSubmit = () => {
      if (!smilesInput.value) {
        ElMessage.warning('Please enter a SMILES expression');
        return;
      }
  
      const encodedSmiles = encodeURIComponent(encodeURIComponent(smilesInput.value));
  
      router.push({
        name: 'compound-prediction',
        query: { 
          smiles: encodedSmiles,
          timestamp: Date.now()
        }
      });
    };
    
    // 清除输入
    const clearInput = () => {
      smilesInput.value = '';
    };
    
    // 使用示例分子
    const useExample = (smiles: string) => {
      smilesInput.value = smiles;
    };

    // 导航函数
    const navigateTo = (path: string) => {
      router.push(path);
    };
    
    const goToNews = () => {
      router.push('/news');
    };
    
    const redirectToNews = (url: string) => {
      window.open(url, '_blank');
    };
    
    const redirectToKinome = () => {
      window.open('https://www.cellsignal.cn/learn-and-support/protein-kinases/human-protein-kinases-overview', '_blank');
    };
    
    return {
      smilesInput,
      exampleMolecules,
      databaseStats,
      drugTypeStats,
      kinaseFamilyData,
      newsList,
      databaseColors,
      drugTypeColors,
      kinaseFamilyColors,
      databaseStatsChart,
      drugTypeChart,
      handleSubmit,
      clearInput,
      useExample,
      navigateTo,
      goToNews,
      redirectToNews,
      redirectToKinome
    };
  }
});
</script>

<style lang="scss" scoped>
// 全局样式重置
* {
  box-sizing: border-box;
}

.homepage-container {
  min-height: 100vh;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  
}

// Hero区域样式
.hero-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 80px 20px;
  text-align: center;
  position: relative;
  overflow: hidden;

  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('../assets/images/home/homeback.png') center/cover;
    opacity: 0.1;
    z-index: 0;
  }

  .hero-content {
    max-width: 1200px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
  }

  .hero-header {
    margin-bottom: 50px;

    .main-title {
      font-size: 4rem;
      font-weight: 700;
      margin-bottom: 10px;
      text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
      letter-spacing: 2px;
    }

    .subtitle {
      font-size: 1.5rem;
      font-weight: 300;
      margin-bottom: 20px;
      opacity: 0.9;
    }

    .description {
      font-size: 1.1rem;
      opacity: 0.8;
      max-width: 600px;
      margin: 0 auto;
      line-height: 1.6;
    }
  }

  .search-container {
    max-width: 800px;
    margin: 0 auto;

    .search-box {
      background: rgba(255, 255, 255, 0.95);
      border-radius: 15px;
      padding: 30px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
      backdrop-filter: blur(10px);

      .search-form {
        .search-input {
          :deep(.el-input__wrapper) {
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            transition: all 0.3s;
            
            &:hover, &.is-focus {
              border-color: #1976D2;
              box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.1);
            }
          }

          :deep(.el-input__inner) {
            font-size: 16px;
            color: #333;
            padding-left: 50px;
          }

          .search-icon {
            color: #1976D2;
            font-size: 18px;
          }
        }

        .search-actions {
          display: flex;
          gap: 15px;
          justify-content: center;
          margin-top: 20px;

          .predict-button {
            background: linear-gradient(45deg, #1976D2, #42A5F5);
            border: none;
            border-radius: 8px;
            padding: 12px 30px;
            font-weight: 600;
            transition: all 0.3s;

            &:hover {
              transform: translateY(-2px);
              box-shadow: 0 5px 15px rgba(25, 118, 210, 0.4);
            }
          }

          .clear-button {
            background: rgba(255, 255, 255, 0.8);
            border: 2px solid #e0e0e0;
            color: #666;
            border-radius: 8px;
            transition: all 0.3s;

            &:hover {
              background: white;
              border-color: #1976D2;
              color: #1976D2;
            }
          }
        }
      }
    }

    .example-molecules {
      margin-top: 25px;
      text-align: center;

      .example-label {
        color: rgba(255, 255, 255, 0.8);
        font-size: 14px;
        margin-right: 15px;
      }

      .molecule-tags {
        display: inline-flex;
        gap: 10px;
        flex-wrap: wrap;
        justify-content: center;

        .molecule-tag {
          background: rgba(255, 255, 255, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.3);
          color: white;
          cursor: pointer;
          transition: all 0.3s;
          border-radius: 20px;

          &:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
          }
        }
      }
    }
  }
}

// 功能模块区域
.features-section {
  padding: 80px 20px;
  background: white;

  .container {
    max-width: 1400px;
    margin: 0 auto;
  }

  .section-title {
    text-align: center;
    font-size: 2.5rem;
    color: #2c3e50;
    margin-bottom: 60px;
    font-weight: 600;
  }

  .features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 30px;

    .feature-card {
      background: white;
      border-radius: 15px;
      padding: 40px 30px;
      text-align: center;
      box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
      transition: all 0.3s;
      cursor: pointer;
      border: 2px solid transparent;

      &:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        border-color: #1976D2;
      }

      .feature-icon {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 25px;
        font-size: 2rem;
        color: white;

        &.family-icon { background: linear-gradient(45deg, #FF6B6B, #FF8E53); }
        &.search-icon { background: linear-gradient(45deg, #4ECDC4, #44A08D); }
        &.prediction-icon { background: linear-gradient(45deg, #667eea, #764ba2); }
        &.drugs-icon { background: linear-gradient(45deg, #f093fb, #f5576c); }
        &.molecule-icon { background: linear-gradient(45deg, #4facfe, #00f2fe); }
        &.about-icon { background: linear-gradient(45deg, #43e97b, #38f9d7); }
      }

      .feature-title {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 15px;
        font-weight: 600;
      }

      .feature-description {
        color: #666;
        line-height: 1.6;
        margin-bottom: 25px;
        font-size: 1rem;
      }

      .feature-stats {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;

        .stat-number {
          font-size: 1.8rem;
          font-weight: 700;
          color: #1976D2;
        }

        .stat-label {
          color: #666;
          font-size: 0.9rem;
        }
      }
    }
  }
}

.statistics-section {
  padding: 80px 20px;
  background: #f8f9fa;

  .container {
    max-width: 1400px;
    margin: 0 auto;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 40px;
    margin-bottom: 60px;

    @media (max-width: 992px) {
      grid-template-columns: 1fr;
      gap: 30px;
    }
  }

  .stats-chart, .drug-stats {
    background: white;
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
    box-sizing: border-box; /* 确保内边距和边框不影响宽度计算 */

    .chart-title {
      font-size: 1.5rem;
      color: #2c3e50;
      margin-bottom: 25px;
      text-align: center;
      font-weight: 600;
    }

    .chart-container {
      height: auto; /* 移除固定高度 */
      aspect-ratio: 16/9; /* 设置宽高比 */
      margin-bottom: 25px;
    }
  }

  .chart-legend {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;

    .legend-item {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.9rem;

      .legend-color {
        width: 12px;
        height: 12px;
        border-radius: 50%;
      }

      .legend-label {
        flex: 1;
        color: #666;
      }

      .legend-value {
        font-weight: 600;
        color: #2c3e50;
      }
    }
  }

  .drug-summary {
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 15px;

      .summary-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px;
        border-radius: 8px;
        background: #f8f9fa;

        .summary-icon {
          width: 40px;
          height: 40px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-size: 1.2rem;
        }

        .summary-content {
          display: flex;
          flex-direction: column;

          .summary-label {
            font-size: 0.8rem;
            color: #666;
          }

          .summary-count {
            font-size: 1.1rem;
            font-weight: 600;
            color: #2c3e50;
          }
        }
      }
    }
  }


  // 激酶分类树状图区域
  .kinome-section {
    background: white;
    border-radius: 15px;
    padding: 40px;
    box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);

    .section-title {
      text-align: center;
      font-size: 2rem;
      color: #2c3e50;
      margin-bottom: 40px;
      font-weight: 600;
    }

    .kinome-content {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 40px;
      align-items: start;

      @media (max-width: 992px) {
        grid-template-columns: 1fr;
        gap: 30px;
      }

      .kinome-image-container {
        text-align: center;

        .kinome-image {
          max-width: 100%;
          height: auto;
          border-radius: 10px;
          cursor: pointer;
          transition: transform 0.3s;
          box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);

          &:hover {
            transform: scale(1.05);
          }
        }
      }

      .kinome-info {
        .kinome-subtitle {
          font-size: 1.3rem;
          color: #2c3e50;
          margin-bottom: 25px;
          font-weight: 600;
        }

        .family-grid {
          display: flex;
          flex-direction: column;
          gap: 15px;

          .family-item {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 10px;
            border-radius: 8px;
            background: #f8f9fa;

            .family-name {
              flex: 1;
              font-size: 0.9rem;
              color: #2c3e50;
              font-weight: 500;
            }

            .family-count {
              font-size: 0.9rem;
              font-weight: 600;
              color: #1976D2;
              min-width: 30px;
            }

            .family-bar {
              flex: 1;
              height: 8px;
              background: #e0e0e0;
              border-radius: 4px;
              overflow: hidden;

              .family-progress {
                height: 100%;
                border-radius: 4px;
                transition: width 0.3s;
              }
            }
          }
        }
      }
    }
  }
}

// 新闻区域
.news-section {
  padding: 80px 20px;
  background: white;

  .container {
    max-width: 1400px;
    margin: 0 auto;
  }

  .news-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 50px;

    .section-title {
      font-size: 2.5rem;
      color: #2c3e50;
      font-weight: 600;
    }

    .more-button {
      color: #1976D2;
      font-size: 1.1rem;
      font-weight: 500;

      &:hover {
        color: #1565C0;
      }
    }
  }

  .news-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 30px;

    .news-card {
      background: white;
      border-radius: 15px;
      overflow: hidden;
      box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
      transition: all 0.3s;
      cursor: pointer;

      &:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);

        .news-overlay {
          opacity: 1;
        }
      }

      .news-image {
        position: relative;
        height: 200px;
        overflow: hidden;

        img {
          width: 100%;
          height: 100%;
          object-fit: cover;
          transition: transform 0.3s;
        }

        .news-overlay {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(25, 118, 210, 0.8);
          display: flex;
          align-items: center;
          justify-content: center;
          opacity: 0;
          transition: opacity 0.3s;

          i {
            color: white;
            font-size: 2rem;
          }
        }

        &:hover img {
          transform: scale(1.1);
        }
      }

      .news-content {
        padding: 25px;

        .news-meta {
          display: flex;
          justify-content: space-between;
          margin-bottom: 15px;
          font-size: 0.9rem;
          color: #666;

          .news-source {
            font-weight: 500;
            color: #1976D2;
          }
        }

        .news-title {
          font-size: 1.2rem;
          color: #2c3e50;
          margin-bottom: 15px;
          font-weight: 600;
          line-height: 1.4;
        }

        .news-summary {
          color: #666;
          line-height: 1.6;
          font-size: 0.95rem;
        }
      }
    }
  }
}

// 关于数据库区域
.about-section {
  padding: 80px 20px;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);

  .container {
    max-width: 1400px;
    margin: 0 auto;
  }

  .about-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 60px;
    align-items: center;

    @media (max-width: 992px) {
      grid-template-columns: 1fr;
      gap: 40px;
    }

    .about-text {
      .about-title {
        font-size: 2.5rem;
        color: #2c3e50;
        margin-bottom: 25px;
        font-weight: 600;
      }

      .about-description {
        font-size: 1.1rem;
        color: #666;
        line-height: 1.8;
        margin-bottom: 30px;
      }

      .about-features {
        display: flex;
        flex-direction: column;
        gap: 15px;

        .about-feature {
          display: flex;
          align-items: center;
          gap: 15px;
          font-size: 1rem;
          color: #555;

          i {
            color: #1976D2;
            font-size: 1.2rem;
          }
        }
      }
    }

    .about-visual {
      text-align: center;

      .kinase-illustration {
        img {
          max-width: 100%;
          height: auto;
          border-radius: 15px;
          box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
      }
    }
  }
}

// 响应式设计
@media (max-width: 768px) {
  .hero-section {
    padding: 60px 20px;

    .hero-header .main-title {
      font-size: 3rem;
    }

    .search-container .search-box {
      padding: 20px;
    }
  }

  .features-section,
  .statistics-section,
  .news-section,
  .about-section {
    padding: 60px 20px;
  }

  .section-title {
    font-size: 2rem !important;
  }

  .features-grid {
    grid-template-columns: 1fr !important;
  }

  .news-grid {
    grid-template-columns: 1fr !important;
  }
}
</style>

