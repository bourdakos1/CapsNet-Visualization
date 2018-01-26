import React, { Component } from 'react'
import SidebarItem from './SidebarItem'
import './Sidebar.css'

class Sidebar extends Component {
  state = {
    activeItem: 0
  }

  onItemClick = (name, index) => {
    this.props.onInputClick(name)
    this.setState({
      activeItem: index
    })
  }

  render() {
    const { inputs } = this.props
    return (
      <div>
        <div className="Sidebar">
          {inputs.map((item, i) => (
            <SidebarItem
              onItemClick={this.onItemClick}
              name={item}
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
