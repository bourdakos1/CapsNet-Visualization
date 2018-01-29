import React, { Component } from 'react'
import Photo from './Photo'
import RangeSlider from './RangeSlider'
import './Reconstruction.css'

class Reconstruction extends Component {
  state = {
    vector: [
      0.19660855,
      0.20227766,
      -0.1382982,
      -0.31342947,
      -0.16363921,
      0.19339905,
      -0.13441671,
      0.43849992,
      0.25891876,
      -0.09146166,
      0.28226382,
      -0.29621538,
      -0.35972677,
      0.20608754,
      -0.15802761,
      0.12676048
    ],
    predicted: 9
  }

  renderNewValues = () => {
    fetch('/api/reconstruct', {
      method: 'POST',
      credentials: 'same-origin',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        vector: this.state.vector,
        predicted: this.state.predicted
      })
    })
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
        this.setState({
          file: response.url
        })
      })
  }

  handleChange = (index, value) => {
    var vector = this.state.vector
    vector[index] = parseFloat(value)
    this.setState(
      {
        vector: vector
      },
      () => {
        fetch('/api/reconstruct', {
          method: 'POST',
          credentials: 'same-origin',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            vector: this.state.vector,
            predicted: this.state.predicted
          })
        })
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
            this.setState({
              file: response.url
            })
          })
      }
    )
  }

  render() {
    return (
      <div className="Reconstruction-container">
        <div className="Reconstruction">
          {this.state.vector.map((item, i) => (
            <RangeSlider
              index={i}
              default={item}
              onValueChange={this.handleChange}
            />
          ))}
        </div>
        <div className="Reconstruction-right">
          <Photo filePath={this.state.file || this.props.files[0]} />
        </div>
      </div>
    )
  }
}

export default Reconstruction
