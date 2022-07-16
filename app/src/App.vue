<template>
  <div id="app">
  <br>
  <h1 class="title">Automated Analysis of Cyber Attack Reports</h1>
  <br>
  <h2>Enter the name of a cyber attack to predict the tactics and techniques (TTPs) used according to MITRE's Attack Framework</h2>
  <br><br>
  <b-input placeholder="Search..." rounded v-model="searchBar"></b-input>
  <br>
  <h2>or enter directly a report to retrieve its corresponding TTPs</h2>
  <br>
  <b-field>
    <b-input type="textarea"
      minlength="10"
      maxlength="100"
      placeholder="Submit Report"
      v-model="searchTextArea">
    </b-input> 
  </b-field>
  <b-button type="is-link is-light" rounded>Predict !</b-button><br />
  </div>
</template>

<script>

export default {
  name: 'App',
  data() {
    return {
      searchBar: '',
      searchTextArea: '',
    }
  },
  methods: {
    submitSearchForm() {
      alert(this.searchBar)
    },
    submitTextForm(){
      alert(this.searchTextArea)
    }
  },

}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  /* -webkit-font-smoothing: antialiased; */ /* this has been causing errors for some reason? */
  /* -moz-osx-font-smoothing: grayscale; */
  text-align: center;
  color:midnightblue;
}
h1.title{
  font-size: 50px;
  font-family: Trebuchet MS;
  color: midnightblue;
}
p{
  color: #666e77;
}
#nav {
  padding: 30px;
}
button{
  margin: 0 10px; 
  padding: 10px; 
  border: none; 
  border-radius: 4px; 
  color: midnightblue;
}
</style>
