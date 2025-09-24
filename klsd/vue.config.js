

const { defineConfig } = require('@vue/cli-service')
const webpack = require('webpack')

module.exports = defineConfig({
  transpileDependencies: true,
  publicPath: process.env.NODE_ENV === 'production' ? '/' : './',
  assetsDir: '',
  filenameHashing: false,
  
  configureWebpack: {
    plugins: [
      new webpack.DefinePlugin({
        'process.env': {
          VUE_APP_API_BASE_URL: JSON.stringify(
            process.env.NODE_ENV === 'production' 
              ? 'http://ai.njucm.edu.cn:8889' 
              : 'http://localhost:8889'
          ),
          __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: JSON.stringify(false),
          __VUE_OPTIONS_API__: JSON.stringify(true),
          __VUE_PROD_DEVTOOLS__: JSON.stringify(false)
        }
      })
    ]
  },
  devServer: {
    historyApiFallback: true,  // 添加这行
    proxy: {
      '/api': {
        target: 'http://localhost:8889',  // 注意这里应该是8889不是8888
        changeOrigin: true,
        pathRewrite: {
          '^/api': ''
        }
      }
    }
  }
})
