import React, { Component } from 'react'
import './SidebarItem.css'

class SidebarItem extends Component {
  onItemClick = () => {
    this.props.onItemClick(this.props.index)
  }

  render() {
    const { active, name } = this.props
    return (
      <div
        onClick={this.onItemClick}
        className={`SidebarItem-container ${active && 'SidebarItem-active'}`}
      >
        <div className="SidebarItem">
          <svg width="16px" height="16px" viewBox="0 0 12 16" version="1.1">
            <g id="file-media" fill="#9da5b4">
              <path
                d="M6,5 L8,5 L8,7 L6,7 L6,5 L6,5 Z M12,4.5 L12,14 C12,14.55 11.55,15 11,15 L1,15 C0.45,15 0,14.55 0,14 L0,2 C0,1.45 0.45,1 1,1 L8.5,1 L12,4.5 L12,4.5 Z M11,5 L8,2 L1,2 L1,13 L4,8 L6,12 L8,10 L11,13 L11,5 L11,5 Z"
                id="Shape"
              />
            </g>
          </svg>
          <div className="SidebarItem-text">{name}</div>
        </div>
      </div>
    )
  }
}

export default SidebarItem
