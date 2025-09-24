<template>
  <div class="prediction-app">
    <!-- 顶部标题 -->
    <div class="header-section">
      <h1>Predict JAK inhibitors</h1>
    </div>

    <!-- 输入区域 -->
    <main class="app-main">
      <div class="input-card card">
        <div class="card-header">
          <h2><i class="fas fa-pencil-alt"></i> SMILES Input</h2>
        </div>
        <div class="card-body">
          <div class="smiles-input-group">
            <!-- SMILES 输入框 -->
            <div class="input-with-label">
              <label for="smiles-input"></label>
              <div class="input-with-button">
                <input
                  id="smiles-input"
                  v-model="smilesInput"
                  placeholder="e.g. CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
                  @keyup.enter="predict"
                />
                <button 
                  class="predict-btn always-visible"
                  @click="predict"
                  :disabled="loading"
                >
                  <i class="fas fa-bolt"></i> Predict
                </button>
              </div>
            </div>

            <!-- 快捷操作 -->
            <div class="quick-actions">
              <button 
                class="clear-btn"
                @click="clear"
              >
                <i class="fas fa-eraser"></i> Clear
              </button>

              <!-- 示例分子 -->
              <div class="examples-section">
                <div class="examples-label">Example Molecules:</div>
                <div class="examples-buttons">
                  <button
                    v-for="(smiles, name) in exampleCompounds"
                    :key="name"
                    class="example-btn"
                    @click="loadExample(name)"
                  >
                    {{ name }} <span class="smiles-hint">{{ smiles }}</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 结果区域 -->
      <div class="result-card card" v-if="result && !loading">
        <div class="card-header">
          <h2><i class="fas fa-chart-bar"></i> Prediction Results</h2>
          <div class="timestamp">
            <i class="far fa-clock"></i> {{ predictionTime }}
          </div>
        </div>

        <div class="card-body">
          <!-- 分子信息 -->
          <div class="molecule-info">
            <div class="smiles-display">
              <label>Current Molecule:</label>
              <div class="smiles-value">{{ result.compound }}</div>
            </div>
          </div>

          <!-- 分子可视化 -->
          <div class="molecule-viewer" v-if="smilesInput && !loading">
            <div class="loading-placeholder" v-if="!molImage">
              <i class="fas fa-atom fa-spin"></i> Generating molecular structure...
            </div>
            <img 
              v-else
              :src="molImage" 
              :alt="'Structure: ' + smilesInput"
              class="molecule-img"
            />
          </div>
          <!-- 整体活性预测 -->
          <div class="activity-prediction" v-if="result.allPrediction">
            <h3 class="section-title">Activity Prediction</h3>
            
            <div class="target-card" :class="result.allPrediction.is_active[0] ? 'active' : 'inactive'">
              <div class="target-header">
                <h3>SMILES Activity</h3>
                <div class="activity-value" :class="result.allPrediction.is_active[0] ? 'active' : 'inactive'">
                  {{ result.allPrediction.predicted_activity?.[0]?.toFixed(2) ?? 'N/A' }} pAct
                </div>
              </div>
              
              <div class="target-body">
                <div class="conclusion" :class="result.allPrediction.is_active[0] ? 'active' : 'inactive'">
                  {{ result.allPrediction.is_active[0] ? 'Active' : 'Inactive' }}
                </div>
                
                <div class="confidence-meter">
                  <div 
                    class="confidence-bar"
                    :style="{ width: getConfidenceWidth(result.allPrediction.predicted_activity?.[0] || 0) }"
                  ></div>
                  <div class="confidence-labels">
                    <span>Weak</span>
                    <span>Medium</span>
                    <span>Strong</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- 激酶活性预测 -->
          <div class="activity-prediction" v-if="result.activityPrediction">
            <h3 class="section-title">Prediction of Kinase Activity Value</h3>
            <div class="activity-value">
              <div class="activity-label">Predicted pAct Value:</div>
              <div class="activity-number">
                {{ result.activityPrediction !== undefined 
                  ? result.activityPrediction.toFixed(2) 
                  : 'N/A' 
                }}
              </div>
            </div>
          </div>

          <!-- 激酶抑制预测 -->
          <div class="inhibition-prediction">
            <h3 class="section-title">Prediction of Kinase Inhibition</h3>
            <div class="results-grid">
              <template v-for="target in targetKinases" :key="target">
                <div 
                  class="target-card"
                  :class="{
                    'active': result.predictions[target]?.predictedActivity >= 6,
                    'inactive': result.predictions[target]?.predictedActivity < 6
                  }"
                >
                  <div class="target-header">
                    <h3>{{ target }}</h3>
                    <div class="activity-badge">
                      {{ result.predictions[target]?.predictedActivity?.toFixed(2) || 'N/A' }} pAct
                    </div>
                  </div>
                  <div class="target-body">
                    <div class="conclusion">
                      {{ result.predictions[target]?.predictedActivity >= 6 ? 'Active' : 'Inactive' }}
                    </div>
                    <div class="confidence-meter">
                      <div 
                        class="confidence-bar"
                        :style="{ width: getConfidenceWidth(result.predictions[target]?.predictedActivity) }"
                      ></div>
                      <div class="confidence-labels">
                        <span>Weak</span>
                        <span>Medium</span>
                        <span>Strong</span>
                      </div>
                    </div>
                  </div>
                </div>
              </template>
            </div>
          </div>

          <!-- 模型对比表格 -->
          <div class="model-comparison card" v-if="result.modelComparison && filteredTargets.length > 0">
            <div class="card-header">
              <h2><i class="fas fa-table"></i> Model Comparison Results</h2>
            </div>
            <div class="card-body">
              <div class="comparison-table-container">
                <table class="comparison-table">
                  <thead>
                    <tr>
                      <th>Target</th>
                      <th v-for="model in modelColumns" :key="model">
                        {{ model.toUpperCase() }}
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="target in filteredTargets" :key="target">
                      <td>{{ target }}</td>
                      <td v-for="model in modelColumns" :key="model"
                          :class="safeGetPredictionClass(result.modelComparison[target][model])">
                        {{ safeGetPredictionText(result.modelComparison[target][model]) }}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 加载状态 -->
      <div class="loading-overlay" v-if="loading">
        <div class="spinner">
          <i class="fas fa-atom fa-spin"></i>
          <p>Predicting...</p>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import axios from 'axios';
import { useRoute } from 'vue-router';
import { useToast } from 'vue-toast-notification';
import 'vue-toast-notification/dist/theme-sugar.css';

export default {
  name: 'CompoundPrediction',

  setup() {
    const route = useRoute();
    const toast = useToast();
    return { route, toast };
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
      apiBaseUrl: process.env.VUE_APP_API_BASE_URL || 'http://localhost:8889',
      vaeApiBaseUrl: process.env.VUE_APP_VAE_API_BASE_URL || 'http://localhost:8889',
      gnnApiBaseUrl: process.env.VUE_APP_GNN_API_BASE_URL || 'http://localhost:8889',
      tdtApiBaseUrl: process.env.VUE_APP_TDT_API_BASE_URL || 'http://localhost:8889',
      cnnApiBaseUrl: process.env.VUE_APP_CNN_API_BASE_URL || 'http://localhost:8889',
      allApiBaseUrl: process.env.VUE_APP_ALL_API_BASE_URL || 'http://localhost:8889',
      modelColumns: [
        'svm', 'knn', 'xgboost', 'rf',
        'cnn', 'gcn', 'gat', 'rgcn',
        'gcn_vae', 'gat_vae', 'rgcn_vae'
      ],
      targetKinases: ['JAK1', 'JAK2', 'JAK3', 'TYK2']
    };
  },

  computed: {
    filteredTargets() {
      if (!this.result?.modelComparison) return [];
      return this.targetKinases.filter(target => 
        this.result.modelComparison[target] !== undefined
      );
    }
  },

  created() {
    this.getSmilesFromRoute();
  },

  watch: {
    '$route': 'getSmilesFromRoute'
  },

  methods: {
    getSmilesFromRoute() {
      try {
        const smilesParam = this.route.query.smiles;
        if (smilesParam) {
          this.smilesInput = decodeURIComponent(decodeURIComponent(smilesParam));
          if (this.validateSmiles(this.smilesInput)) {
            this.$nextTick(this.predict);
          }
        }
      } catch (err) {
        console.error('SMILES参数解析失败:', err);
        this.showToast('SMILES参数无效', 'error');
      }
    },

    validateSmiles(smiles) {
      return typeof smiles === 'string' && 
             smiles.length > 3 && 
             /^[a-zA-Z0-9@+\-[\]()/=#$]+$/.test(smiles);
    },

    async predict() {
      if (!this.smilesInput.trim()) {
        this.showToast('请输入有效的SMILES表达式', 'error');
        return;
      }

      this.loading = true;
      this.resetState();

      try {
        const imageResponse = await this.fetchMoleculeImage();
        this.molImage = imageResponse.data.imageUrl;

        const [mainPrediction, allPrediction, vaePrediction, gnnPrediction, tdtPrediction, cnnPrediction] = await Promise.all([
          this.fetchMainPrediction(),
          this.fetchAllPrediction(),
          this.fetchVaePrediction(this.molImage),
          this.fetchGnnPrediction(this.molImage),
          this.fetchTdtPrediction(),
          this.fetchCnnPrediction(this.molImage)
        ]);

        this.handleCombinedResponse(mainPrediction, allPrediction, vaePrediction, gnnPrediction, tdtPrediction, cnnPrediction);
      } catch (err) {
        this.handleError(err);
      } finally {
        this.loading = false;
      }
    },

    async fetchMainPrediction() {
      try {
        const response = await axios.post(
          `${this.apiBaseUrl}/api/predict`,  // 主预测端点
          { smiles: this.smilesInput },
          {
            headers: { 
              'Content-Type': 'application/json',
              'Accept': 'application/json'
            },
            timeout: 30000
          }
        );
        console.log('分靶点预测响应:', response.data); // 调试日志
        if (!response?.data) {
          throw new Error('主预测接口返回空数据');
        }
        return response.data;
      } catch (error) {
        console.error('主预测请求失败:', error);
        throw new Error(`主模型预测失败: ${this.extractErrorMessage(error)}`);
      }
    },

    async fetchAllPrediction() {
      try {
        const response = await axios.post(
          `${this.allApiBaseUrl}/api/all/predict`,
          { smiles: this.smilesInput },
          { timeout: 30000 }
        );

        console.log('整体预测原始响应:', JSON.stringify(response.data, null, 2));

        // 修正判断条件（注意两层data结构）
        if (!response?.data?.data?.prediction) {
          console.warn('预测数据为空，使用默认值');
          return {
            data: {  // 保持外层data结构
              prediction: {
                predicted_activity: 0,
                is_active: false
              }
            }
          };
        }

        // 返回修正后的结构
        return {
          data: {
            prediction: response.data.data.prediction
          }
        };

      } catch (error) {
        console.error('请求失败:', error);
        return { 
          data: {
            prediction: {
              predicted_activity: 0,
              is_active: false
            }
          }
        };
      }
    },

    async fetchVaePrediction(imageUrl) {
      try {
        const response = await axios.post(
          `${this.vaeApiBaseUrl}/api/vae-predict`,
          { 
            smiles: this.smilesInput,
            image_url: imageUrl 
          },
          {
            headers: { 
              'Content-Type': 'application/json',
              'Accept': 'application/json'
            },
            timeout: 30000
          }
        );

        if (!response?.data) {
          throw new Error('VAE接口返回空数据');
        }
        return response.data;
      } catch (error) {
        console.error('VAE预测请求失败:', error);
        return { data: { vae_predictions: {} } };
      }
    },

    async fetchGnnPrediction(imageUrl) {
      try {
        const response = await axios.post(
          `${this.gnnApiBaseUrl}/api/gnn-predict`,
          { 
            smiles: this.smilesInput,
            image_url: imageUrl 
          },
          {
            headers: { 
              'Content-Type': 'application/json',
              'Accept': 'application/json'
            },
            timeout: 30000
          }
        );

        if (!response?.data) {
          throw new Error('GNN接口返回空数据');
        }
        return response.data;
      } catch (error) {
        console.error('GNN预测请求失败:', error);
        return { data: { gnn_predictions: {} } };
      }
    },

    async fetchTdtPrediction() {
      try {
        const response = await axios.post(
          `${this.tdtApiBaseUrl}/api/tdt-predict`,
          { smiles: this.smilesInput },
          {
            headers: { 
              'Content-Type': 'application/json',
              'Accept': 'application/json'
            },
            timeout: 30000
          }
        );

        if (!response?.data) {
          throw new Error('传统模型接口返回空数据');
        }
        return response.data;
      } catch (error) {
        console.error('传统模型预测请求失败:', error);
        return { data: { tdt_predictions: {} } };
      }
    },

    async fetchCnnPrediction(imageUrl) {
      try {
        const response = await axios.post(
          `${this.cnnApiBaseUrl}/api/cnn-predict`,
          { 
            smiles: this.smilesInput,
            image_url: imageUrl 
          },
          {
            headers: { 
              'Content-Type': 'application/json',
              'Accept': 'application/json'
            },
            timeout: 30000
          }
        );

        if (!response?.data) {
          throw new Error('CNN接口返回空数据');
        }
        return response.data;
      } catch (error) {
        console.error('CNN预测请求失败:', error);
        return { data: { cnn_predictions: {} } };
      }
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
          throw new Error('分子图像生成失败');
        }
        return response;
      } catch (error) {
        console.error('分子图像请求失败:', error);
        throw new Error('分子结构生成失败');
      }
    },

    handleCombinedResponse(mainData, allData, vaeData, gnnData, tdtData, cnnData) {
      try {
        console.log('整体预测处理数据:', allData); // 调试日志
        console.log('分靶点预测数据:', mainData); // 调试日志
            // 调试日志
        console.log('allData 结构:', JSON.stringify(allData.data, null, 2));
        // 修正数据访问路径
        const backendPrediction = allData.data?.prediction || {
          predicted_activity: 0,
          is_active: false
        };

        // 转换为前端需要的数组格式
        const allPrediction = {
          predicted_activity: [backendPrediction.predicted_activity],
          is_active: [backendPrediction.is_active]
        };

        // 2. 验证并处理主预测数据
        if (!mainData?.data?.predictions) {
          throw new Error('主预测数据缺少 predictions 字段');
        }

        // 3. 处理分靶点预测结果
        const predictions = {};
        Object.entries(mainData.data.predictions).forEach(([target, data]) => {
          predictions[target] = {
            predictedActivity: data.predicted_activity,
            conclusion: data.predicted_activity >= 6 ? 'Active' : 'Inactive',
            confidenceWidth: this.getConfidenceWidth(data.predicted_activity)
          };
        });

        // 4. 处理模型比较结果
        const modelComparison = {};
        this.targetKinases.forEach(target => {
          modelComparison[target] = {};
          const lowerTarget = target.toLowerCase();

          // 主模型结果
          if (mainData.data.model_comparison?.[target]) {
            modelComparison[target] = {
              ...mainData.data.model_comparison[target]
            };
          }

          // VAE模型结果
          const vaePredictions = vaeData.data?.vae_predictions || {};  // 修正路径
          if (vaePredictions[target]) {  // 用大写target匹配（如"JAK1"）
            modelComparison[target] = {
              ...modelComparison[target],
              // 映射模型名称（后端大写 -> 前端小写+下划线）
              gcn_vae: vaePredictions[target].GCN_VAE,
              gat_vae: vaePredictions[target].GAT_VAE,
              rgcn_vae: vaePredictions[target].RGCN_VAE
            };
          }

          // GNN模型结果
          const gnnPredictions = gnnData.data?.gnn_predictions || {};
          if (gnnPredictions[lowerTarget]) {
            modelComparison[target] = {
              ...modelComparison[target],
              gcn: gnnPredictions[lowerTarget].gcn,
              gat: gnnPredictions[lowerTarget].gat,
              rgcn: gnnPredictions[lowerTarget].rgcn
            };
          }

          // 传统模型结果
          const tdtPredictions = tdtData?.tdt_predictions?.tdt_predictions || {};
          if (tdtPredictions[lowerTarget]) {
            modelComparison[target] = {
              ...modelComparison[target],
              svm: tdtPredictions[lowerTarget].svm,
              knn: tdtPredictions[lowerTarget].knn,
              xgboost: tdtPredictions[lowerTarget].xgboost,
              rf: tdtPredictions[lowerTarget].rf
            };
          }

          // CNN模型结果
          const cnnPredictions = cnnData?.data?.cnn_predictions?.cnn_predictions || {};
          if (cnnPredictions[lowerTarget]) {
            modelComparison[target] = {
              ...modelComparison[target],
              cnn: cnnPredictions[lowerTarget]
            };
          }
        });

        // 5. 构建最终结果对象
        this.result = {
          compound: this.smilesInput,
          allPrediction: allPrediction,
          predictions: predictions,
          modelComparison: modelComparison,
          activityPrediction: mainData.data?.activity_prediction 
            ? parseFloat(mainData.data.activity_prediction) 
            : 0,
          timestamp: Date.now()
        };

        console.log('最终结果:', this.result); // 调试日志
        this.predictionTime = new Date().toLocaleString();
        this.showToast('预测完成', 'success');
      } catch (error) {
        console.error('响应处理失败:', error);
        this.showToast(`结果处理错误: ${error.message}`, 'error');
        this.result = {
          compound: this.smilesInput,
          allPrediction: {
            predicted_activity: [0],
            is_active: [false]
          },
          predictions: {},
          modelComparison: {},
          activityPrediction: 0,
          timestamp: Date.now()
        };
      }
    },

    safeGetPredictionClass(modelData) {
      try {
        if (!modelData) return 'error';
        if (modelData.prediction === 'active') return 'active';
        if (modelData.prediction === 'inactive') return 'inactive';
        if (modelData.probability !== undefined) {
          return modelData.probability >= 0.5 ? 'active' : 'inactive';
        }
        if (modelData.error) return 'error';
        return 'error';
      } catch (e) {
        console.error('safeGetPredictionClass错误:', e);
        return 'error';
      }
    },

    safeGetPredictionText(modelData) {
      try {
        if (!modelData) return 'Error';
        if (modelData.prediction) return modelData.prediction.charAt(0).toUpperCase() + modelData.prediction.slice(1);
        if (modelData.probability !== undefined) {
          return modelData.probability >= 0.5 ? 'Active' : 'Inactive';
        }
        if (modelData.error) return 'Error';
        return 'N/A';
      } catch (e) {
        console.error('safeGetPredictionText错误:', e);
        return 'Error';
      }
    },

    extractErrorMessage(error) {
      if (error.response) {
        return error.response.data?.error || 
               error.response.data?.message || 
               `HTTP ${error.response.status}`;
      }
      return error.message || '未知错误';
    },

    handleError(err) {
      console.error('预测流程错误:', err);
      let message = '预测失败';
      if (err.message.includes('主模型')) message = '主模型预测失败';
      else if (err.message.includes('VAE')) message = 'VAE模型预测失败';
      else if (err.message.includes('GNN')) message = 'GNN模型预测失败';
      else if (err.message.includes('TDT')) message = 'TDT模型预测失败';
      else if (err.message.includes('CNN')) message = 'CNN模型预测失败';
      else if (err.message.includes('分子图像')) message = '分子结构生成失败';

      this.showToast(`${message}: ${this.extractErrorMessage(err)}`, 'error');
      this.resetState();
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
        duration: 5000
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
      if (!activity) return '0%';
      // 调整范围以适应您的活性值范围
      const minValue = 4;  // 假设最小活性值
      const maxValue = 10; // 假设最大活性值
      
      // 标准化到0-100%范围
      const normalized = Math.min(maxValue, Math.max(minValue, activity)) - minValue;
      return `${(normalized / (maxValue - minValue)) * 100}%`;
    }
  }
};
</script>

<style scoped>
/* Base Styles - 颜色变量调整为首页蓝紫色系 */
/* 统一后的颜色变量 - 与首页保持一致 */
:root {
  --primary: #1976D2; /* 恢复为首页的蓝色 */
  --primary-light: #42A5F5;
  --secondary: #764ba2; /* 与首页的紫色渐变保持一致 */
  --success: #4CAF50;
  --danger: #F44336;
  --warning: #FF9800;
  --info: #00BCD4;
  --light: #f5f7fa; /* 与首页背景一致 */
  --dark: #2c3e50; /* 与首页文字颜色一致 */
  --gradient-primary:  linear-gradient(135deg, #667eea 0%, #764ba2 100%); /* 首页hero区域渐变 */
}

.prediction-app {
  width: 100%;
  margin: 0;
  padding: 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #333;
  /* 新增：与首页背景风格统一 */
  background: linear-gradient(135deg,rgb(255, 255, 255) 0%,rgb(206, 222, 241) 100%);
  min-height: 100vh;
}

/* 新增：通过伪元素叠加背景图 */
.prediction-app::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  /* 替换为你的背景图路径 */
  background: url('D:\desktop\YYQ\KLSD\klsd\src\assets\images\home\homeback.png') center/cover;
  opacity: 0.08; /* 图片透明度（0.1表示10%透明） */
  z-index: 0; /* 确保图片在背景色之上，内容之下 */
  pointer-events: none; /* 避免图片干扰交互 */
}

/* 所有内容容器添加z-index，确保在背景图之上 */
.header-section, .card, .results-grid, .model-comparison, .activity-prediction {
  position: relative;
  z-index: 1;
}

/* Header Section */
.header-section {
  margin-bottom: 50px;
  padding: 0 20px;
}

.header-section h1 {
  color: #1976D2; /* 使用统一主色 */
  font-size: 2.1rem;
  margin: 0;
  padding: 0;
  font-weight: 700;
  letter-spacing: 0.5px;
}

/* Card Styles - 卡片头部渐变与首页hero区域呼应 */
.card {
  background: white;
  border-radius: 12px;
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1); /* 与首页卡片阴影风格一致 */
  margin-bottom: 30px;
  overflow: hidden;
  width: 100%;
  transition: all 0.3s;
  border: 2px solid transparent;
}

.card:hover {
  transform: translateY(-5px);
  box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15); /* 与首页悬停效果一致 */
}

.card-header {
  padding: 15px 25px;
  /* 替换粉色渐变为首页蓝紫色渐变 */
  background:  linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
  opacity: 0.8;
}

.card-header h2 {
  margin: 0;
  font-size: 1.7rem;
  font-weight: 500;
  display: flex;
  align-items: center;
  text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2); /* 与首页文字阴影一致 */
}

.card-header h2 i {
  margin-right: 12px;
  font-size: 1.2em;
}

.card-body {
  padding: 25px;
}

/* Input Area - 聚焦样式与首页搜索框统一 */
.smiles-input-group {
  margin-bottom: 40px;
}

.input-with-label label {
  display: block;
  margin-bottom: 20px;
  font-weight: 600;
  color: var(--dark);
  font-size: 1.3rem;
}

.input-with-button {
  display: flex;
  margin-bottom: 30px;
  align-items: stretch;
}

.input-with-button input {
  flex: 1;
  padding: 25px;
  border: 2px solid #e0e0e0; /* 与首页输入框一致 */
  border-radius: 10px 0 0 10px; /* 圆角与首页一致 */
  font-size: 1.2rem;
  border-right: none;
  transition: all 0.3s;
}

.input-with-button input:focus {
  outline: none;
  border-color: #e0e0e0;
  /* 替换粉色阴影为首页蓝色阴影 */
  box-shadow: 0 0 0 3px rgba(25, 118, 210, 0.2);
}

/* 预测按钮（强制始终显示 + 悬停仅加深颜色） */
.predict-btn.always-visible {
  padding: 0 30px;
  background:linear-gradient(135deg,rgb(102, 155, 234) 0%,rgb(143, 109, 165) 100%);
  color: white;
  border: 1px solid #667eea;
  border-radius: 0 10px 10px 0; /* 圆角与首页一致 */
  cursor: pointer;
  font-weight: 600;
  font-size: 1.3rem;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 140px;
  transition: all 0.3s;
  opacity: 1 !important;
  visibility: visible !important;
}

.predict-btn.always-visible:hover {
  background:linear-gradient(135deg,rgb(123, 118, 187) 100%); /* Darker pink for hover */
  border-color: #667eea;
  transform: translateY(-2px); /* 与首页按钮悬停效果一致 */
  box-shadow: 0 5px 15px rgba(25, 118, 210, 0.4); /* 与首页一致 */
}

.predict-btn.always-visible:disabled {
  background: #cccccc;
  border-color: #cccccc;
  cursor: not-allowed;
}
/* Quick Actions - 保持风格，颜色与主色协调 */
.quick-actions {
  display: flex;
  flex-direction: column;
  gap: 20px;
  margin-top: 10px;
}

.clear-btn {
  height: 63px;
  padding: 0px 8px;
  width: 130px; /* 固定宽度 */
  min-width: unset; /* 移除最小宽度限制 */
  background:linear-gradient(135deg,rgb(102, 155, 234) 0%,rgb(143, 109, 165) 100%);
  color: white;
  border: 1px solid #667eea;
  border-radius:  10px; /* 圆角与首页一致 */
  cursor: pointer;
  font-weight: 600;
  font-size: 1.3rem;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s;
  opacity: 1 !important;
  visibility: visible !important;
}


.clear-btn:hover {
  background:linear-gradient(135deg,rgb(123, 118, 187) 100%); /* Darker pink for hover */
  border-color: #667eea;
  transform: translateY(-2px); /* 与首页按钮悬停效果一致 */
  box-shadow: 0 5px 15px rgba(25, 118, 210, 0.4); /* 与首页一致 */
}

.clear-btn i {
  margin-right: 8px;
}

/* Examples Section - 示例按钮颜色与首页标签统一 */
.examples-section {
  background: #f8f9fa;
  border-radius: 12px;
  padding: 20px;
  margin-top: 20px;
  height:160px;
}

.examples-label {
  font-weight: 600;
  margin-bottom: 25px;
  color: var(--dark);
  font-size: 1.3rem;
}

.examples-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}

.example-btn {
  padding: 15px 29px;
  /* 替换浅粉色背景为首页标签风格的浅蓝色 */
  background:rgb(209, 220, 230);
  color: var(--primary);
  border: 1px solid #e0e0e0; /* 与首页输入框一致 */
  border-radius: 10px; /* 与首页标签圆角一致 */
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s; /* 与首页标签过渡一致 */
  min-width: 250px;
  text-align: left;
  font-weight: 550;
}

.example-btn:hover {
  /* 替换深粉色hover为首页标签hover风格 */
  background: #bbdefb;
  transform: translateY(-2px); /* 与首页标签hover上移一致 */
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.smiles-hint {
  font-size: 0.9rem;
  color: #666;
  display: block;
  margin-top: 8px;
  font-family: monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* Molecule Viewer - 保持布局，微调背景与首页协调 */
.molecule-viewer {
  margin-top: 30px;
  text-align: center;
  padding: 30px;
  background: #f9f9f9;
  border-radius: 12px;
  min-height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
}

.loading-placeholder {
  color: var(--secondary); /* 使用辅助色 */
  font-size: 1.3rem;
  display: flex;
  align-items: center;
}

.loading-placeholder i {
  margin-right: 15px;
  font-size: 1.5em;
  animation: spin 1s linear infinite;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
.molecule-img {
  max-width: 100%;
  max-height: 400px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
  border-radius: 8px;
}
.timestamp {
  font-size: 1.3rem; /* 日期字体大小 */
  margin-left: 18px; /* 与标题的横向间距，可调整 */
  color: #f9f9f9; /* 可选，调整日期颜色让层级更清晰 */
}


/* Target Card Styles - 活跃状态颜色与首页主色呼应 */
.target-card {
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
  transition: all 0.3s;
  background: white;
}

.target-card:hover {
  transform: translateY(-10px); /* 与首页卡片悬停效果一致 */
  box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
}

.target-card.active {
  border-top: 4px solid var(--success);
}

.target-card.inactive {
  border-top: 4px solid var(--danger);
}

.target-header {
  padding: 20px;
  background: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #eee;
}

.target-header h3 {
  margin: 0;
  color: var(--dark);
  font-size: 1.3rem;
  font-weight: 600;
}

.activity-badge {
  padding: 8px 16px;
  background: var(--light);
  border-radius: 20px;
  font-weight: bold;
  font-size: 0.9rem;
}

.active .activity-badge {
  /* 替换绿色背景为首页主色浅色背景 */
  background: rgba(25, 118, 210, 0.1);
  color: var(--primary);
}

.inactive .activity-badge {
  background: #ffebee;
  color: var(--danger); /* 保持不变 */
}

.target-body {
  padding: 20px;
  background: #f9f9f9;
}

.conclusion {
  font-size: 1.3rem;
  font-weight: bold;
  margin-bottom: 20px;
  text-align: center;
}

.active .conclusion {
  color: var(--primary); /* 替换绿色为主题蓝 */
}

.inactive .conclusion {
  color: var(--danger); /* 保持不变 */
}

/* Confidence Meter - 渐变与首页风格呼应 */
.confidence-meter {
  margin-top: 25px;
}

.confidence-bar {
  height: 12px;
  /* 调整渐变起始色为首页蓝紫色 */
  background: linear-gradient(90deg,rgb(226, 160, 206),rgb(141, 69, 214),rgb(240, 58, 58));
  border-radius: 6px;
  margin-bottom: 10px;
  overflow: hidden;
}

.confidence-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.9rem;
  color: #666;
}

/* Model Comparison - 活跃状态颜色与首页统一 */
<style scoped>
/* 修改 Model Comparison Results 卡片样式 */
/* Model Comparison - 卡片样式与其他部分统一 */
.model-comparison.card {
  margin-top: 30px;
}

.model-comparison .card-header {
  /* 保持与其他卡片头部一致的渐变 */
  background: linear-gradient(135deg,rgb(102, 155, 234) 0%,rgb(143, 109, 165) 100%);
}

.model-comparison .card-header h2 {
  color: white;
  font-size: 1.7rem;
  margin: 0;
}

.model-comparison .card-body {
  padding: 25px;
}

/* 表格容器 */
.comparison-table-container {
  margin: 0;
  overflow-x: auto;
  background: white;
  border-radius: 10px;
  padding: 1rem;
}

/* 表格主体 */
.comparison-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: 1.1rem;
}

/* 表头单元格 */
.comparison-table th {
  color: black;
  padding: 12px 15px;
  font-weight: 500;
  text-align: center;
  background-color: #f8f9fa;
}

/* 表格单元格 */
.comparison-table td {
  padding: 12px 15px;
  border: 1px solid #e0e0e0;
  text-align: center;
  background: white;
}

/* 活性状态单元格 */
.comparison-table .active {
  background: rgba(25, 118, 210, 0.1);
  color:rgb(64, 84, 197);
  font-weight: 600;
}

/* 非活性状态单元格 */
.comparison-table .inactive {
  background: rgba(244, 67, 54, 0.1);
  color:rgb(228, 72, 72);
  font-weight: 600;
}

.comparison-table .error {
  background-color: #fff8e1;
  color: #f57f17;
}


/* Loading State - 加载图标颜色与首页统一 */
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
  font-size: 3.5rem;
  color: var(--primary); /* 替换粉色为主题蓝 */
  margin-bottom: 20px;
}

.spinner p {
  font-size: 1.5rem;
  color: var(--dark);
}

/* Responsive Adjustments - 保持布局适配 */
@media (max-width: 768px) {
  .header-section {
    padding: 0 15px;
  }
  
  .header-section h1 {
    font-size: 1.4rem;
  }
  
  .card-body {
    padding: 20px 15px;
  }
  
  .results-grid {
    grid-template-columns: 1fr; 
    gap: 20px;
  }
  
  .input-with-button {
    flex-direction: column;
  }
  
  .input-with-button input {
    border-radius: 5px;
    margin-bottom: 10px;
    width: 100%;
  }
  
  .input-with-button button {
    border-radius: 5px;
    width: 100%;
  }
  
  .examples-buttons {
    flex-direction: column;
  }
  
  .example-btn {
    min-width: 100%;
    width: 100%;
  }
  
  .comparison-table th, 
  .comparison-table td {
    padding: 12px 15px;
    font-size: 1rem;
  }
}

/* 新增或修改的样式部分 - 颜色与首页统一 */
.molecule-info .smiles-display label {
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--dark);
}
.molecule-info .smiles-value {
  font-size: 1.3rem; /* 分子表达式字体大小 */
  line-height: 1.4; /* 控制长文本换行后的行间距 */
  margin-top: 10px; /* 与上方标签的间距 */
  color: #444; /* 可选，调整文本颜色 */
}


/* Kinase Activity Prediction Section - 活动值颜色与首页统一 */
.section-title {
  color: #2c3e50; /* 深蓝色文字，与首页统一 */
  font-size: 1.5rem; /* 调大字体 */
  font-weight: 600; /* 加粗 */
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* 统一字体 */
  letter-spacing: 0.5px; /* 字母间距 */
  margin-bottom: 1.3rem; /* 与下方内容的间距 */
  text-transform: capitalize; /* 首字母大写 */
  position: relative; /* 用于添加装饰线 */
  padding-bottom: 15px; /* 装饰线间距 */
}

.activity-prediction {
  margin: 3rem 0; /* 上下3rem，左右0 */
  padding: 2rem; /* 内间距 */
  background: white;
  border-radius: 12px;
  box-shadow: 0 5px 20px rgba(0,0,0,0.08); /* 与卡片统一 */
}

/* 活性/非活性值颜色样式 */
.activity-value {
  font-size: 1.3rem;
  font-weight: bold;
  padding: 8px 16px;
  border-radius: 20px;
  color: #000000; /* 强制文字颜色为黑色 */
}

.activity-value.active {
  background: rgba(25, 118, 210, 0.1); /* 活性浅蓝色背景 */
}

.activity-value.inactive {
  background: rgba(244, 67, 54, 0.1); /* 非活性浅红色背景 */
}

/* 调整卡片头部布局 */
.target-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px;
}

.target-header h3 {
  margin: 0;
  font-size: 1.3rem;
  color: #2c3e50;
}

/* 调整活性徽章样式 */
.activity-badge {
  padding: 8px 16px;
  border-radius: 20px;
  font-weight: bold;
  font-size: 1rem;
  color: #000000; /* 强制文字颜色为黑色 */
}

.activity-badge.active {
  background: rgba(25, 118, 210, 0.1); /* 活性浅蓝色背景 */
}

.activity-badge.inactive {
  background: rgba(244, 67, 54, 0.1); /* 非活性浅红色背景 */
}

/* Kinase Inhibition Section */
/* 激酶抑制预测整体容器 */
.inhibition-prediction {
  /* 与 Activity Prediction 保持相同的外间距、内间距、背景和阴影 */
  margin: 3rem 0; 
  padding: 2rem; 
  background: white;
  border-radius: 12px;
  box-shadow: 0 5px 20px rgba(0,0,0,0.08);
}
/* Results Grid - 保持布局，卡片阴影与首页统一 */
/* 修改：强制results-grid为一行四列 */
.results-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr); /* 固定四列 */
  gap: 15px; /* 减小间距，使整体更紧凑 */
  margin: 20px 0; /* 调整上下间距 */
}

/* 可选：调整单个卡片的最小宽度 */
.target-card {
  min-width: 0; /* 允许卡片缩小 */
}

/* 确保Activity Prediction的卡片宽度与整体一致 */
.activity-prediction .target-card {
  width: 100%;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .activity-value {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .activity-label {
    margin-bottom: 8px;
  }
}
</style>