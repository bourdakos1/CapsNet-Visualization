import React, { Component } from 'react'
import './Tab.css'

class Tab extends Component {
  onTabClick = () => {
    this.props.onTabClick(this.props.index)
  }

  render() {
    const { active, name } = this.props
    return (
      <div
        onClick={this.onTabClick}
        className={`Tab ${active && 'Tab-active'}`}
      >
        {name}
      </div>
    )
  }
}

export default Tab
