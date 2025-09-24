<template>
    <div class="contact-us-container">
      <h1>Contact Us</h1>
      <form @submit.prevent="submitForm">
        <div class="form-group">
          <label for="name">Name:</label>
          <input type="text" id="name" v-model="contact.name" required>
        </div>
        <div class="form-group">
          <label for="email">Email:</label>
          <input type="email" id="email" v-model="contact.email" required>
        </div>
        <div class="form-group">
          <label for="message">Message:</label>
          <textarea id="message" rows="4" v-model="contact.message" required></textarea>
        </div>
        <button type="submit">Send Message</button>
      </form>
    </div>
</template>
    
<script lang="ts">
import { ref, defineComponent } from "vue";
import KlsdBackground from "@/components/layouts/KlsdBackground.vue";
import Captcha from "@/components/layouts/KlsdCaptcha.vue";
import axios from 'axios';
    
export default defineComponent({
    name: 'ContactUsPage',
    components: {

    },
    data() {
      return {
        contact: {
        name: '',
        email: '',
        message: ''
      }
      };
    },
    methods: {
        submitForm() {
      // Here you would handle the form submission,
      // e.g., send the data to a server or validate input
      console.log(this.contact);
      alert('Message sent!');
      // Reset the form after submission
      this.contact.name = '';
      this.contact.email = '';
      this.contact.message = '';
    }
    }
});
</script>
  
<style lang="scss" scoped>
@import "@/assets/css/var.scss";
@import "@/assets/css/global.scss";
  
.contact-us-container {
  max-width: 600px;
  margin: 0 auto;
  padding: 2rem;
  text-align: center;
}

.form-group {
  margin-bottom: 1rem;
  text-align: left;
}

label {
  display: block;
  margin-bottom: 0.5rem;
}

input[type="text"],
input[type="email"],
textarea {
  width: 100%;
  padding: 0.5rem;
  margin-bottom: 0.5rem;
  border: 1px solid #ccc;
  border-radius: 0.25rem;
}

button {
  background: linear-gradient(to right, #667eea, #764ba2); /* 替换为具体的渐变颜色 */
  color: white;
  padding: 0.75rem 1.5rem;
  border: 1px solid #667eea;
  border-radius: 0.25rem;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.3s ease;

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); /* 使用#667eea的rgba值 */
  }
}
</style>
    