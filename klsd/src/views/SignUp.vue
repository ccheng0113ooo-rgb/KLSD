<!-- <template>
  <klsd-login-logo></klsd-login-logo>
  <div class="sign">
    <div class="sign-head">
      <span>Register</span>
    </div>
    <el-form ref="signUpForm" label-width="70px" status-icon :model="registerForm" :rules="SignUpRules">
      <el-form-item prop="name" label="username">
        <el-input v-model="registerForm.name" placeholder="username"></el-input>
      </el-form-item>
      <el-form-item prop="password" label="password">
        <el-input type="password" placeholder="password" v-model="registerForm.password"></el-input>
      </el-form-item>
      <el-form-item prop="sex" label="sex">
        <el-radio-group v-model="registerForm.sex">
          <el-radio :label="0">women</el-radio>
          <el-radio :label="1">men</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item prop="phone" label="telephone">
        <el-input placeholder="telephone" v-model="registerForm.phone"></el-input>
      </el-form-item>
      <el-form-item prop="type" label="industry">
        <el-input v-model="registerForm.type" placeholder="industry"></el-input>
      </el-form-item>
      <el-form-item prop="identity" label="position">
        <el-input v-model="registerForm.identity" placeholder="position"></el-input>
      </el-form-item>
      <el-form-item class="sign-btn">
        <el-button @click="goBack()">Login</el-button>
        <el-button type="primary" @click="handleSignUp(formRef)">Confirm</el-button>
      </el-form-item>
    </el-form>
  </div>
</template>

<script lang="ts">
import { defineComponent, reactive, getCurrentInstance } from "vue";
import mixin from "@/mixins/mixin";
import KlsdLoginLogo from "@/components/layouts/KlsdLoginLogo.vue";
import { HttpManager } from "@/api";
import { RouterName, NavName, SignUpRules } from "@/enums";

export default defineComponent({
  // components: {
  //   KlsdLoginLogo,
  // },
  setup() {
    const { proxy } = getCurrentInstance();
    const { routerManager, changeIndex } = mixin();

    const registerForm = reactive({
      name: "",
      password: "",
      sex: "",
      phone: "",
      type: "",
      identity: "",
    });

    async function handleSignUp() {
      let canRun = true;
      (proxy.$refs["signUpForm"] as any).validate((valid) => {
        if (!valid) return (canRun = false);
      });
      if (!canRun) return;


      try {
        const name = registerForm.name;
        const password = registerForm.password;
        const sex = registerForm.sex;
        const phone = registerForm.phone;
        const type = registerForm.type;
        const identity = registerForm.identity;
        const result = (await HttpManager.SignUp({name,password,sex,phone,type,identity})) as ResponseBody;
        (proxy as any).$message({
          message: result.message,
          type: result.type,
        });

        if (result.success) {
          changeIndex(NavName.SignIn);
          routerManager(RouterName.SignIn, { path: RouterName.SignIn });
        }
      } catch (error) {
        console.error(error);
      }
    }

    return {
      SignUpRules,
      registerForm,
      handleSignUp,
    };
  },
});
</script>

<style lang="scss" scoped>
@import "@/assets/css/sign.scss";
</style> -->
