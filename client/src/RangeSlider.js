import React, { Component } from 'react'
import './RangeSlider.css'

class RangeSlider extends Component {
  state = {
    length: 0,
    start: 0
  }

  onValueChange = e => {
    // THis is pretty ugly
    if (this.state.lock == null) {
      this.setState({
        lock: this.props.default
      })
    }
    var lock = this.state.lock || this.props.default
    var val = e.target.value

    this.props.onValueChange(this.props.index, val)

    var x = val * 100 + 50
    var y = lock * 100 + 50

    this.setState({
      length: Math.abs(x - y),
      start: Math.min(x, y),
      value: val
    })
  }

  render() {
    return (
      <input
        style={{
          background: `linear-gradient(to right, #4b5364 0%, #4b5364 ${this
            .state.start}%, #568af2 ${this.state.start}%, #568af2 ${this.state
            .start + this.state.length}%, #4b5364 ${this.state.start +
            this.state.length}%, #4b5364 100%)`
        }}
        type="range"
        min="-.5"
        max=".5"
        step="0.00000001"
        value={this.state.value || this.props.default}
        className="RangeSlider-range"
        onChange={this.onValueChange}
      />
    )
  }
}

export default RangeSlider
