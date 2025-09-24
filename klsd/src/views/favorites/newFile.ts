import { ref, defineComponent, computed } from "vue";
import mixin from "@/mixins/mixin";
import { HttpManager } from "@/api";

export default defineComponent({
components: {},

setup() {
const { routerManager, changeIndex } = mixin();
const pageSize = ref(1000); // 页数
const currentPage = ref(1); // 当前页
const tableData = ref([]); // 激酶
const input_search = ref('');
// const input_group = ref('');
const input_family = ref('');
const input_pchembl1 = ref('');
const input_pchembl2 = ref('');
const groups = ref([]); // 存储 group 数据
const subfamilies = ref([]); // 存储 subfamily 数据
const target = ref(null); //全局变量
const data = computed(() => tableData.value.slice((currentPage.value - 1) * pageSize.value, currentPage.value * pageSize.value));
// 获取全部激酶
const uniqueGroups = computed(() => {
const groupSet = new Set(groups.value.map(item => item.groupname));
return Array.from(groupSet).map(groupname => ({
id: groupname, // 这里简单地将 groupname 当作 id
groupname,
}));
});

const clearFilters = () => {
window.location.reload();
};

async function getFilteredData(input_pchembl1, input_pchembl2) {
let min;
let max;
let target1;
target1 = 'Name1=' + target.value;
if (input_pchembl1 === null) {
min = null;
} else {
min = '&&pchemblvalue1=' + input_pchembl1;
}
if (input_pchembl2 === null) {
max = null;
} else {
max = '&&pchemblvalue2=' + input_pchembl2;
}
target1 = target1 + min + max;
if (input_pchembl1 === null && input_pchembl2 === null) {
tableData.value = ((await HttpManager.getChemblKinaseOfLikeTargetName(target.value)) as ResponseBody).data;
currentPage.value = 1;
} else {
tableData.value = ((await HttpManager.getChemblKinaseOfLikePChemblValue(target1)) as ResponseBody).data;
currentPage.value = 1;
target1 = null;
}

}

async function getChemblKinase() {
tableData.value = ((await HttpManager.getChemblKinase()) as ResponseBody).data;
currentPage.value = 1;
}

async function getChemblKinaseOfLikeTargetName(input_search) {
tableData.value = ((await HttpManager.getChemblKinaseOfLikeTargetName(input_search)) as ResponseBody).data;
currentPage.value = 1;
}

async function getBrowseList() {
const response = (await HttpManager.getBrowseList()) as ResponseBody;
groups.value = response.data.map(item => ({ id: item.id, groupname: item.groupname }));
subfamilies.value = response.data.map(item => ({ id: item.id, subfamilyname: item.subfamilyname }));
}

try {
getBrowseList();
getChemblKinase();
} catch (error) {
console.error(error);
}

// 获取激酶
async function handleChangeView(item) {
tableData.value = [];
try {
if (input_search.value === null) {
await getChemblKinase();
} else {
await getChemblKinaseOfLikeTargetName(input_search);
}
} catch (error) {
console.error(error);
}
}

// 获取当前页
function handleCurrentChange(val) {
currentPage.value = val;
}
const handleChange = (event) => {
target.value = input_family.value;
console.log('选中的子家族是：', target.value);
};

return {
pageSize,
currentPage,
tableData,
data,
target,
handleChangeView,
handleCurrentChange,
handleChange,
input_search,
// input_group,
input_family,
// input_target: '',
input_pchembl1,
input_pchembl2,
uniqueGroups,
subfamilies,
getChemblKinaseOfLikeTargetName,
getFilteredData,
clearFilters,
};
},
//   methods: {
//     deleteRow (index, rows) {
//       rows.splice(index, 1);
//     },
//     handleSizeChange (val) {
//       console.log(`每页 ${val} 条`);
//     },
//     handleCurrentChange (val) {
//       console.log(`当前页: ${val}`);
//     },
//     getList () {
//     axios.get('http://localhost:8088/favoriteslist', {
//       params: {
//         //pageSize: this.pageSize,
//       },
//     }).then(
//       success => {
//         console.log('请求成功！');
//         console.log(success);
//         this.tableData = success.data;
//       //   this.pageNum = success.data.current;
//       //   this.pageSize = success.data.size;
//       //   this.total = success.data.total;
//         //this.gettotal(this.tableData.length);
//         // this.pageSizeChange();
//         // this.currentPageChange();
//         //this.calculatePageRange();
//       })
//       .catch(error => {
//         console.log('请求失败！');
//         console.log(error.message);
//       }
//       );
//   },
//   },
//   mounted () {
//   this.getList();
//   // this.gettotal(this.tableData);
// },
});
