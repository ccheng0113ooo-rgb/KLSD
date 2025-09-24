import { createStore } from "vuex";
import configure from "./configure";
import user from "./user";
import group from "./group";
import  compound from "./compound";

export default createStore({
  modules: {
    configure,
    user,
    group,
    compound,
  },
});
