<!-- <template>
  <klsd-login-logo></klsd-login-logo>
  <div class="sign">
    <div class="sign-head">
      <span>Login</span>
    </div>
    <el-form ref="signInForm" status-icon :model="registerForm" :rules="SignInRules">
      <el-form-item prop="phone">
        <el-input placeholder="telephone" v-model="registerForm.phone"></el-input>
      </el-form-item>
      <el-form-item prop="password">
        <el-input type="password" placeholder="password" v-model="registerForm.password" @keyup.enter="handleLoginIn"></el-input>
      </el-form-item>
      <el-form-item class="sign-btn">
        <el-button @click="handleSignUp">Register</el-button>
        <el-button type="primary" @click="handleLoginIn">Login</el-button>
      </el-form-item>
    </el-form>
  </div>
</template>

<script lang="ts">
import { defineComponent, reactive, getCurrentInstance } from "vue";
import mixin from "@/mixins/mixin";
import KlsdLoginLogo from "@/components/layouts/KlsdLoginLogo.vue";
import { HttpManager } from "@/api";
import { NavName, RouterName, SignInRules } from "@/enums";

export default defineComponent({
  // components: {
  //   KlsdLoginLogo,
  // },
  setup() {
    const { proxy } = getCurrentInstance();
    const { routerManager, changeIndex } = mixin();

    // 登录用户名密码
    const registerForm = reactive({
      phone: "",
      password: "",
    });

    async function handleLoginIn() {
      let canRun = true;
      (proxy.$refs["signInForm"] as any).validate((valid) => {
        if (!valid) return (canRun = false);
      });
      if (!canRun) return;


      try {
        const phone = registerForm.phone;
        const password = registerForm.password;
        const result = (await HttpManager.signIn({phone,password})) as ResponseBody;
        (proxy as any).$message({
          message: result.message,
          type: result.type,
        });

        if (result.success) {
          proxy.$store.commit("setUserId", result.data[0].id);
          proxy.$store.commit("setUsername", result.data[0].name);
          // proxy.$store.commit("setUserPic", result.data[0].avator);
          proxy.$store.commit("setToken", true);
          changeIndex(NavName.Home);
          routerManager(RouterName.Home, { path: RouterName.Home });
        }
      } catch (error) {
        console.error(error);
      }
    }

    function handleSignUp() {
      routerManager(RouterName.SignUp, { path: RouterName.SignUp });
    }

    return {
      registerForm,
      SignInRules,
      handleLoginIn,
      handleSignUp,
    };
  },
});
</script>

<style lang="scss" scoped>
@import "@/assets/css/sign.scss";
</style> -->
