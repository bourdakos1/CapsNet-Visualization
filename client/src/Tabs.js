import React, { Component } from 'react'
import Tab from './Tab'
import './Tabs.css'

class Tabs extends Component {
  state = {
    activeTab: 0
  }

  onTabClick = index => {
    this.props.onTabClick(index)
    this.setState({
      activeTab: index
    })
  }

  render() {
    var tabs = [
      { name: 'Input' },
      { name: 'Conv1 Kernel' },
      { name: 'Conv1 + ReLU' },
      { name: 'Primary Caps' },
      { name: 'Digit Caps' },
      { name: 'Reconstruction' }
    ]
    return (
      <div>
        <div className="Tabs">
          {tabs.map((tab, i) => (
            <Tab
              onTabClick={this.onTabClick}
              index={i}
              name={tab.name}
              active={i === this.state.activeTab}
            />
          ))}
        </div>
      </div>
    )
  }
}

export default Tabs
