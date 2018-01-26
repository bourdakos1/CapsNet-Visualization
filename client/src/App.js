import React, { Component } from 'react'
import Tabs from './Tabs'
import Sidebar from './Sidebar'
import PhotoGrid from './PhotoGrid'
import './App.css'

class App extends Component {
  state = {
    bulk: {},
    inputs: [],
    activeInput: '',
    activeTab: 0
  }

  componentDidMount() {
    this.getImages()
  }

  getImages = () => {
    return fetch('/images')
      .then(response => {
        if (response.status >= 200 && response.status < 300) {
          return response
        }
        const error = new Error('Failed to load')
        throw error
      })
      .then(response => {
        return response.json().then(response => {
          if (response.error != null) {
            const error = new Error(response.error)
            throw error
          }
          return response
        })
      })
      .then(response => {
        var inputs = []
        for (var input in response) {
          inputs = [...inputs, input]
        }
        this.setState({
          bulk: response,
          inputs: inputs,
          activeInput: inputs[0],
          activeTab: 0
        })
      })
  }

  onInputClick = input => {
    this.setState({
      activeInput: input
    })
  }

  onTabClick = index => {
    this.setState({
      activeTab: index
    })
  }

  render() {
    const { activeInput, activeTab, bulk } = this.state
    var files = []
    if (
      bulk != null &&
      activeInput in bulk &&
      bulk[activeInput].length > activeTab
    ) {
      files = bulk[activeInput][activeTab]
    }

    return (
      <div className="App">
        <Tabs onTabClick={this.onTabClick} />
        <Sidebar onInputClick={this.onInputClick} inputs={this.state.inputs} />
        <PhotoGrid files={files} />
      </div>
    )
  }
}

export default App
