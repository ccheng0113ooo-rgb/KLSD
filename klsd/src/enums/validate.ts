// 登录规则
const validatePhone = (rule, value, callback) => {
  if (!value) {
    return callback(new Error("The telephone number cannot be null."));
  } else {
    callback();
  }
};

export const validatePassword = (rule, value, callback) => {
  if (value === "") {
    callback(new Error("The password cannot be null."));
  } else {
    callback();
  }
};

export const SignInRules = {
  phone: [{ validator: validatePhone, trigger: "blur", min: 3 }],
  password: [{ validator: validatePassword, trigger: "blur", min: 3 }],
};

// 注册规则
export const SignUpRules = {
  name: [{ required: true, trigger: 'blur', min: 3 }],
  password: [{ required: true, trigger: 'blur', min: 3 }],
  sex: [{ required: true, message: 'Please choose gender.', trigger: 'change' }],
  type: [{ message: 'Please choose industry.', trigger: 'blur' }],
  phone: [
    { message: 'Please input telephone number.', trigger: 'blur' },
    {
      type: 'phone',
      message: 'Please input the correct telephone number.',
      trigger: ['blur', 'change'],
    },
  ],
  age: [{ required: true, message: 'Please choose age.', trigger: 'change' }],
  identity: [{ message: 'Please enter your position.', trigger: 'blur' }],
};
