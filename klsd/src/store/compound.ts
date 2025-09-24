export default {
    state: {
      compoundId: "", // ID
      compoundname: "", // compound名称
      targetname1: "", // target名称1
      targetname2: "", // target名称2
      diffmin: "", // target差值下限
      diffmax: "", // target差值上线
    },
    getters: {
      compoundId: (state) => state.compoundId,
      compoundname: (state) => state.compoundname,
      targetname1: (state) => state.targetname1,
      targetname2: (state) => state.targetname2,
      diffmin: (state) => state.diffmin,
      diffmax: (state) => state.diffmax,
    },
    mutations: {
      setcompoundId: (state, compoundId) => {
        state.compoundId = compoundId;
      },
      setcompoundname: (state, compoundname) => {
        state.compoundname = compoundname;
      },
      settargetname1: (state, targetname1) => {
        state.targetname1 = targetname1;
      },
      settargetname2: (state, targetname2) => {
        state.targetname2 = targetname2;
      },
      setdiffmin: (state, diffmin) => {
        state.diffmin = diffmin;
      },
      setdiffmax: (state, diffmax) => {
        state.diffmax = diffmax;
      },
    },
  };