import React, { Component } from 'react'
import Tabs from './Tabs'
import Body from './Body'
import './App.css'

class App extends Component {
  render() {
    return (
      <div className="App">
        <Tabs />
        <Body />
      </div>
    )
  }
}

export default App
