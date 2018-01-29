import React, { Component } from 'react'
import Photo from './Photo'
import './PhotoGrid.css'

class PhotoGrid extends Component {
  state = {
    activeIndex: 0
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
          {this.props.files.map((path, i) => (
            <GridImage
              active={this.state.activeIndex === i}
              onImageClick={this.onImageClick}
              path={path}
              index={i}
            />
          ))}
        </div>
        <div className="PhotoGrid-right">
          <Photo filePath={this.props.files[this.state.activeIndex]} />
        </div>
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
