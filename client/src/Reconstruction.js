import React, { Component } from 'react'
import Photo from './Photo'
import RangeSlider from './RangeSlider'
import './Reconstruction.css'

class Reconstruction extends Component {
  state = {
    file: null,
    vector: [],
    initialVector: [],
    prediction: -1
  }

  componentDidMount() {
    this.getInitialValue()
  }

  componentDidUpdate(prevProps, prevState) {
    if (prevProps.json === this.props.json) {
      return
    }
    this.getInitialValue()
  }

  getInitialValue = () => {
    return fetch(this.props.json)
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
          file: null,
          vector: response['vector'],
          initialVector: response['vector'],
          prediction: response['prediction']
        })
      })
  }

  handleChange = (index, value) => {
    const vector = [...this.state.vector]
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
            predicted: this.state.prediction
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
              value={item}
              default={this.state.initialVector[i]}
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
