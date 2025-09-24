export default {
    state: {
      groupId: "", // ID
      groupName: "", // group
      familyName: "", // family
      subfamilyName: "", // subfamily
      number: 0, // number
      active: 0, // active
      inactive: 0, // inactive
    },
    getters: {
      groupId: (state) => state.groupId,
      groupName: (state) => state.groupName,
      familyName: (state) => state.familyName,
      subfamilyName: (state) => state.subfamilyName,
      number: (state) => state.number,
      active: (state) => state.active,
      inactive: (state) => state.inactive,
    },
    mutations: {
      setGroupId: (state, groupId) => {
        state.groupId = groupId;
      },
      setGroupName: (state, groupName) => {
        state.groupName = groupName;
      },
      setFamilyName: (state, familyName) => {
        state.familyName = familyName;
      },
      setSubfamilyName: (state, subfamilyName) => {
        state.subfamilyName = subfamilyName;
      },
      setNumber: (state, number) => {
        state.number = number;
      },
      setActive: (state, active) => {
        state.active = active;
      },
      setInactive: (state, inactive) => {
        state.inactive = inactive;
      },
    },
  };
  