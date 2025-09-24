<template>
  <div class="prediction-app">
    <!-- 头部区域 -->
    <header class="app-header">
      <h1>
        <i class="fas fa-flask"></i> 化合物活性预测系统
      </h1>
      <p class="subtitle">基于神经网络模型的JAK抑制剂预测平台</p>
    </header>

    <!-- 主内容区 -->
    <main class="app-main">
      <!-- 输入卡片 -->
      <div class="input-card card">
        <div class="card-header">
          <h2><i class="fas fa-pencil-alt"></i> 分子输入</h2>
        </div>
        <div class="card-body">
          <div class="smiles-input-group">
            <div class="input-with-label">
              <label for="smiles-input">SMILES 表达式</label>
              <div class="input-with-button">
                <input
                  id="smiles-input"
                  v-model="smilesInput"
                  placeholder="例如: CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
                  @keyup.enter="predict"
                />
                <button 
                  class="predict-btn"
                  @click="predict"
                  :disabled="loading || !smilesInput.trim()"
                >
                  <i class="fas fa-bolt"></i> 预测
                </button>
              </div>
            </div>
            
            <div class="quick-actions">
              <button 
                class="clear-btn"
                @click="clear"
              >
                <i class="fas fa-eraser"></i> 清除
              </button>
              
              <div class="examples-section">
                <div class="examples-label">示例分子：</div>
                <div class="examples-buttons">
                  <button
                    v-for="(smiles, name) in exampleCompounds"
                    :key="name"
                    class="example-btn"
                    @click="loadExample(name)"
                  >
                    <span class="example-name">{{ name }}</span>
                    <span class="smiles-hint">{{ truncateSmiles(smiles) }}</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
          
          <!-- 分子可视化 -->
          <div class="molecule-viewer" v-if="smilesInput && !loading">
            <div class="loading-placeholder" v-if="!molImage">
              <i class="fas fa-atom fa-spin"></i> 生成分子结构...
            </div>
            <img 
              v-else
              :src="molImage" 
              :alt="'结构图: ' + smilesInput"
              class="molecule-img"
            />
          </div>
        </div>
      </div>

      <!-- 结果卡片 -->
      <div class="result-card card" v-if="result && !loading">
        <div class="card-header">
          <h2><i class="fas fa-chart-bar"></i> 预测结果</h2>
          <div class="timestamp">
            <i class="far fa-clock"></i> {{ predictionTime }}
          </div>
        </div>
        
        <div class="card-body">
          <!-- 分子信息 -->
          <div class="molecule-info">
            <div class="smiles-display">
              <label>当前分子:</label>
              <div class="smiles-value">{{ result.compound }}</div>
            </div>
          </div>
          
          <!-- 主要结果展示 -->
          <div class="results-grid">
            <!-- 靶点预测卡片 -->
            <div 
              v-for="(prediction, target) in result.predictions" 
              :key="target"
              class="target-card"
              :class="{
                'inhibitor': prediction.conclusion === 'Inhibitor',
                'non-inhibitor': prediction.conclusion === 'Non-inhibitor'
              }"
            >
              <div class="target-header">
                <h3>{{ target.toUpperCase() }}</h3>
                <div class="activity-badge">
                  {{ prediction.predictedActivity.toFixed(2) }} pIC50
                </div>
              </div>
              
              <div class="target-body">
                <div class="conclusion">
                  {{ prediction.conclusion }}
                </div>
                
                <div class="confidence-meter">
                  <div 
                    class="confidence-bar"
                    :style="{ width: getConfidenceWidth(prediction.predictedActivity) }"
                  ></div>
                  <div class="confidence-labels">
                    <span>弱</span>
                    <span>中</span>
                    <span>强</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- 模型信息 -->
          <div class="model-info">
            <h4><i class="fas fa-cogs"></i> 模型信息</h4>
            <div class="model-badges">
              <span class="model-badge">SVM</span>
              <span class="model-badge">随机森林</span>
              <span class="model-badge">XGBoost</span>
              <span class="model-badge">深度神经网络</span>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 加载状态 -->
      <div class="loading-overlay" v-if="loading">
        <div class="spinner">
          <i class="fas fa-atom fa-spin"></i>
          <p>预测计算中...</p>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import axios from 'axios';
import { useToast } from 'vue-toast-notification';
import 'vue-toast-notification/dist/theme-sugar.css';

export default {
  setup() {
    const toast = useToast();
    return { toast };
  },
  data() {
    return {
      smilesInput: '',
      loading: false,
      result: null,
      molImage: null,
      predictionTime: null,
      exampleCompounds: {
        'Compound 1': 'COc1ccc(NC(=O)N2CCN3C(=O)c4ccccc4C23c2ccc(Cl)cc2)cc1',
        'Compound 2': 'Cc1ccncc1C(=O)N1CCN(C(=O)C(=O)c2c[nH]c3ccccc23)CC1',
        'Compound 3': 'N#C[C@H]1CCOC[C@@H]1n1cc(C(N)=O)c(Nc2ccc(C3CC3)cc2)n1'
      },
      apiBaseUrl: process.env.VUE_APP_API_BASE_URL || 'http://localhost:8889'
    };
  },
  methods: {
    async predict() {
      if (!this.smilesInput.trim()) {
        this.showToast('请输入有效的SMILES', 'error');
        return;
      }

      this.loading = true;
      this.resetState();

      try {
        const [predictionResponse, imageResponse] = await Promise.all([
          this.fetchPrediction(),
          this.fetchMoleculeImage()
        ]);
        
        this.handleResponse(predictionResponse);
        this.molImage = imageResponse.data.imageUrl;
        this.predictionTime = new Date().toLocaleString();
      } catch (err) {
        this.handleError(err);
      } finally {
        this.loading = false;
      }
    },
    
    async fetchPrediction() {
      try {
        const response = await axios.post(
          `${this.apiBaseUrl}/api/predict`,
          { smiles: this.smilesInput },
          {
            headers: { 'Content-Type': 'application/json' },
            timeout: 30000
          }
        );
        
        console.log('API响应数据:', response.data); // 调试日志
        
        if (!response?.data?.data) {
          throw new Error('无效的API响应格式');
        }
        
        return response;
      } catch (error) {
        console.error('API请求失败:', error);
        if (error.response) {
          console.error('响应数据:', error.response.data);
        }
        throw error;
      }
    },

    handleResponse(response) {
      console.log('完整API响应:', response.data); // 调试日志
      
      const data = response.data.data;
      const predictions = {};
      
      // 明确处理四个目标
      ['jak1', 'jak2', 'jak3', 'tyk2'].forEach(target => {
        if (data[target]?.dnn?.predicted_activity !== undefined) {
          const activity = Number(data[target].dnn.predicted_activity);
          predictions[target] = {
            predictedActivity: activity,
            conclusion: activity >= 6 ? 'Inhibitor' : 'Non-inhibitor'
          };
        } else {
          console.warn(`目标 ${target} 缺少预测数据`);
          predictions[target] = {
            predictedActivity: 0,
            conclusion: 'Error'
          };
        }
      });

      this.result = {
        compound: this.smilesInput,
        predictions,
        activityPrediction: data.activity_prediction || 0,
        modelComparison: data.model_comparison || {},
        timestamp: response.data.timestamp || Date.now()
      };
      
      console.log('处理后结果:', this.result); // 调试日志
    },
    
    async fetchMoleculeImage() {
      try {
        const response = await axios.post(
          `${this.apiBaseUrl}/api/generate-image`,
          { smiles: this.smilesInput },
          {
            headers: { 'Content-Type': 'application/json' },
            timeout: 15000
          }
        );
        
        if (!response?.data?.imageUrl) {
          throw new Error('无效的图像响应');
        }
        
        return response;
      } catch (error) {
        console.error('获取分子图像失败:', error);
        return { data: { imageUrl: null } };
      }
    },
    
    handleError(err) {
      console.error('预测错误:', err);
      
      let userMessage = '预测失败';
      if (err.response) {
        switch (err.response.status) {
          case 400: userMessage = '无效的SMILES格式'; break;
          case 500: userMessage = '服务器错误，请稍后再试'; break;
          case 404: userMessage = 'API端点未找到'; break;
          default: userMessage = `请求失败 (${err.response.status})`;
        }
      } else if (err.request) {
        userMessage = '网络错误，请检查连接';
      } else {
        userMessage = err.message || userMessage;
      }
      
      this.showToast(userMessage, 'error');
    },
    
    resetState() {
      this.result = null;
      this.molImage = null;
    },
    
    showToast(message, type = 'success') {
      this.toast.open({
        message,
        type,
        position: 'top-right',
        duration: 3000
      });
    },
    
    clear() {
      this.smilesInput = '';
      this.resetState();
    },
    
    loadExample(name) {
      if (this.exampleCompounds[name]) {
        this.smilesInput = this.exampleCompounds[name];
        this.$nextTick(this.predict);
      }
    },
    
    getConfidenceWidth(activity) {
      const normalized = Math.min(7, Math.max(0, activity));
      return `${(normalized / 7) * 100}%`;
    }
  }
};
</script>

<style scoped>
/* 基础样式 */
:root {
  --primary: #4a6fa5;
  --secondary: #6b8cae;
  --success: #4caf50;
  --danger: #f44336;
  --warning: #ff9800;
  --info: #00bcd4;
  --light: #f8f9fa;
  --dark: #343a40;
}

.prediction-app {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #333;
}

/* 头部样式 */
.app-header {
  text-align: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid #eee;
}

.app-header h1 {
  color: var(--primary);
  font-size: 2.2rem;
  margin-bottom: 5px;
}

.app-header .subtitle {
  color: var(--secondary);
  font-size: 1.1rem;
  font-weight: 300;
}

/* 卡片通用样式 */
.card {
  background: white;
  border-radius: 10px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  margin-bottom: 25px;
  overflow: hidden;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
  transform: translateY(-5px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
}

.card-header {
  padding: 15px 20px;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header h2 {
  margin: 0;
  font-size: 1.3rem;
  display: flex;
  align-items: center;
}

.card-header h2 i {
  margin-right: 10px;
}

.card-body {
  padding: 20px;
}

/* 输入区域样式 */
.smiles-input-group {
  margin-bottom: 20px;
}

.input-with-label label {
  display: block;
  margin-bottom: 8px;
  font-weight: 600;
  color: var(--dark);
}

.input-with-button {
  display: flex;
  margin-bottom: 15px;
}

.input-with-button input {
  flex: 1;
  padding: 12px 15px;
  border: 1px solid #ddd;
  border-radius: 5px 0 0 5px;
  font-size: 1rem;
  border-right: none;
  transition: border-color 0.3s;
}

.input-with-button input:focus {
  outline: none;
  border-color: var(--primary);
  box-shadow: 0 0 0 2px rgba(74, 111, 165, 0.2);
}

/* 预测按钮样式 */
.predict-btn {
  padding: 0 20px;
  background: var(--primary);
  color: white;
  border: 1px solid var(--primary);
  border-radius: 0 5px 5px 0;
  cursor: pointer;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 90px;
  transition: all 0.2s;
}

.predict-btn:hover:not(:disabled) {
  background: #3a5a8c;
  border-color: #3a5a8c;
}

.predict-btn:disabled {
  background: #cccccc;
  border-color: #cccccc;
  cursor: not-allowed;
  opacity: 0.7;
}

/* 快速操作区域 */
.quick-actions {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.clear-btn {
  padding: 8px 15px;
  background: var(--light);
  border: 1px solid #ddd;
  border-radius: 5px;
  cursor: pointer;
  display: flex;
  align-items: center;
  transition: all 0.3s;
  width: fit-content;
}

.clear-btn:hover {
  background: #e9ecef;
}

.clear-btn i {
  margin-right: 5px;
}

/* 示例分子区域 */
.examples-section {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 12px;
  margin-top: 10px;
}

.examples-label {
  font-weight: 600;
  margin-bottom: 8px;
  color: var(--dark);
}

.examples-buttons {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 8px;
}

.example-btn {
  padding: 10px;
  background: #e3f2fd;
  color: var(--primary);
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s;
  text-align: left;
  display: flex;
  flex-direction: column;
}

.example-btn:hover {
  background: #bbdefb;
  transform: translateY(-2px);
}

.example-name {
  font-weight: 500;
  margin-bottom: 2px;
}

.smiles-hint {
  font-size: 0.75rem;
  color: #666;
  font-family: monospace;
  word-break: break-all;
}

/* 分子可视化区域 */
.molecule-viewer {
  margin-top: 20px;
  text-align: center;
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px dashed #ddd;
}

.loading-placeholder {
  color: var(--secondary);
  font-size: 1.1rem;
}

.loading-placeholder i {
  margin-right: 10px;
}

.molecule-img {
  max-width: 100%;
  max-height: 300px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* 结果网格布局 */
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 20px;
  margin: 25px 0;
}

/* 靶点卡片样式 */
.target-card {
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s;
}

.target-card:hover {
  transform: translateY(-5px);
}

.target-card.inhibitor {
  border-top: 4px solid var(--success);
}

.target-card.non-inhibitor {
  border-top: 4px solid var(--danger);
}

.target-header {
  padding: 15px;
  background: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.target-header h3 {
  margin: 0;
  color: var(--dark);
  font-size: 1.1rem;
}

.activity-badge {
  padding: 5px 10px;
  background: var(--light);
  border-radius: 20px;
  font-weight: bold;
  font-size: 0.9rem;
}

.inhibitor .activity-badge {
  background: #e8f5e9;
  color: var(--success);
}

.non-inhibitor .activity-badge {
  background: #ffebee;
  color: var(--danger);
}

.target-body {
  padding: 15px;
  background: #f9f9f9;
}

.conclusion {
  font-size: 1.1rem;
  font-weight: bold;
  margin-bottom: 15px;
  text-align: center;
}

.inhibitor .conclusion {
  color: var(--success);
}

.non-inhibitor .conclusion {
  color: var(--danger);
}

/* 置信度指示器 */
.confidence-meter {
  margin-top: 15px;
}

.confidence-bar {
  height: 10px;
  background: linear-gradient(90deg, #ff5252, #ff9800, #4caf50);
  border-radius: 5px;
  margin-bottom: 5px;
}

.confidence-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
  color: #666;
}

/* 模型信息 */
.model-info {
  margin-top: 30px;
  padding-top: 20px;
  border-top: 1px solid #eee;
}

.model-info h4 {
  color: var(--dark);
  margin-bottom: 15px;
  display: flex;
  align-items: center;
}

.model-info h4 i {
  margin-right: 10px;
}

.model-badges {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.model-badge {
  padding: 5px 12px;
  background: #e3f2fd;
  color: var(--primary);
  border-radius: 20px;
  font-size: 0.85rem;
}

/* 加载状态 */
.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.spinner {
  text-align: center;
}

.spinner i {
  font-size: 3rem;
  color: var(--primary);
  margin-bottom: 15px;
}

.spinner p {
  font-size: 1.2rem;
  color: var(--dark);
}

/* 响应式调整 */
@media (max-width: 768px) {
  .app-header h1 {
    font-size: 1.8rem;
  }
  
  .results-grid {
    grid-template-columns: 1fr;
  }
  
  .input-with-button {
    flex-direction: column;
  }
  
  .input-with-button input {
    border-radius: 5px;
    margin-bottom: 5px;
    border-right: 1px solid #ddd;
  }
  
  .input-with-button button {
    border-radius: 5px;
    width: 100%;
  }
  
  .examples-buttons {
    grid-template-columns: 1fr;
  }
  
  .card-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .timestamp {
    margin-top: 5px;
    font-size: 0.9rem;
  }
}

/* 动画效果 */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.result-card {
  animation: fadeIn 0.5s ease-out;
}

.target-card {
  animation: fadeIn 0.3s ease-out;
}
</style>