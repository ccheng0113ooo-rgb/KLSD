<!-- <template>
  <el-form ref="updateForm" label-width="70px" :model="registerForm" :rules="SignUpRules">
    <el-form-item prop="username" label="username">
      <el-input v-model="registerForm.name" placeholder="username"></el-input>
    </el-form-item>
    <el-form-item label="sex">
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
    <el-form-item>
      <el-button @click="goBack(-1)">Cancel</el-button>
      <el-button type="primary" @click="saveMsg()">Save</el-button>
    </el-form-item>
  </el-form>
</template>

<script lang="ts">
import { defineComponent, computed, onMounted, getCurrentInstance, reactive } from "vue";
import { useStore } from "vuex";
import mixin from "@/mixins/mixin";
import { SignUpRules } from "@/enums";
import { HttpManager } from "@/api";

export default defineComponent({
  setup() {
    const { proxy } = getCurrentInstance();
    const store = useStore();
    const { goBack } = mixin();

    // 注册
    const registerForm = reactive({
      name: "",
      sex: "",
      phone: "",
      type: "",
      identity: "",
    });
    const userId = computed(() => store.getters.userId);

    async function getUserInfo(id) {
      const result = (await HttpManager.getUserOfId(id)) as ResponseBody;
      registerForm.name = result.data[0].name;
      registerForm.sex = result.data[0].sex;
      registerForm.phone = result.data[0].phone;
      registerForm.type = result.data[0].type;
      registerForm.identity = result.data[0].identity;
    }

    async function saveMsg() {
      let canRun = true;
      (proxy.$refs["updateForm"] as any).validate((valid) => {
        if (!valid) return (canRun = false);
      });
      if (!canRun) return;


      const id = userId.value;
      const name = registerForm.name;
      const sex = registerForm.sex;
      const phone = registerForm.phone;
      const type = registerForm.type;
      const identity = registerForm.identity;
      const result = (await HttpManager.updateUserMsg({id,name,sex,phone,type,identity})) as ResponseBody;
      (proxy as any).$message({
        message: result.message,
        type: result.type,
      });
      if (result.success) {
        proxy.$store.commit("setUsername", registerForm.name);
        goBack(-1);
      }
    }

    onMounted(() => {
      getUserInfo(userId.value);
    });

    return {
      registerForm,
      SignUpRules,
      saveMsg,
      goBack,
    };
  },
});
</script>

<style lang="scss" scoped>
.btn ::v-deep .el-form-item__content {
  display: flex;
  justify-content: center;
}
</style> -->
