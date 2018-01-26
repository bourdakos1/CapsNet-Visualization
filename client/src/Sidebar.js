import React, { Component } from 'react'
import SidebarItem from './SidebarItem'
import './Sidebar.css'

class Sidebar extends Component {
  state = {
    activeItem: 0
  }

  onItemClick = index => {
    this.setState({
      activeItem: index
    })
  }

  render() {
    var items = [
      { name: '0.png' },
      { name: '1.png' },
      { name: '2.png' },
      { name: '3.png' }
    ]
    return (
      <div>
        <div className="Sidebar">
          {items.map((item, i) => (
            <SidebarItem
              onItemClick={this.onItemClick}
              name={item.name}
              active={i === this.state.activeItem}
              index={i}
            />
          ))}
        </div>
        <div className="Sidebar-gap" />
      </div>
    )
  }
}

export default Sidebar
