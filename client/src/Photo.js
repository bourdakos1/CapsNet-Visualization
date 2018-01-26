import React, { Component } from 'react'
import './Photo.css'

class Photo extends Component {
  render() {
    const { filePath } = this.props
    return (
      <div className="Photo">
        <img src={filePath} />
      </div>
    )
  }
}

export default Photo
