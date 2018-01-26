import React, { Component } from 'react'
import Photo from './Photo'
import './PhotoGrid.css'

class PhotoGrid extends Component {
  state = {
    layers: [[]],
    files: [],
    activeIndex: 0,
    activeLayer: 0
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
        console.log(response)
        // this.setState({
        //   files: response.files
        // })
      })
  }

  onImageClick = i => {
    this.setState({
      activeIndex: i
    })
  }

  render() {
    return (
      <div className="PhotoGrid-container">
        <div className="PhotoGrid">
          {this.state.files.map((path, i) => (
            <GridImage
              active={this.state.activeIndex === i}
              onImageClick={this.onImageClick}
              path={`/visualizations/0_conv1_kernel/${path}`}
              index={i}
            />
          ))}
        </div>
        <Photo
          filePath={`/visualizations/0_conv1_kernel/${this.state.files[
            this.state.activeIndex
          ]}`}
        />
      </div>
    )
  }
}

class GridImage extends Component {
  onImageClick = () => {
    const { index } = this.props
    this.props.onImageClick(index)
  }

  render() {
    const { path, active } = this.props
    return (
      <img
        className={`${active && 'PhotoGrid-active'}`}
        onClick={this.onImageClick}
        src={path}
      />
    )
  }
}

export default PhotoGrid
