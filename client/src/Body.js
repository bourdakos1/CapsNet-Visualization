import React, { Component } from 'react'
import Sidebar from './Sidebar'
import PhotoGrid from './PhotoGrid'
import './Body.css'

class Body extends Component {
  render() {
    return (
      <div className="Body">
        <Sidebar />
        <PhotoGrid />
      </div>
    )
  }
}

export default Body
