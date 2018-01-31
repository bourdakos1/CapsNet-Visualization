import React, { Component } from 'react'
import './RangeSlider.css'

class RangeSlider extends Component {
  onValueChange = e => {
    var val = e.target.value
    this.props.onValueChange(this.props.index, val)
  }

  render() {
    var lock = this.props.default
    var val = this.props.value

    var x = val * 100 + 50
    var y = lock * 100 + 50

    var length = Math.abs(x - y)
    var start = Math.min(x, y)

    return (
      <input
        style={{
          background: `linear-gradient(to right, #4b5364 0%, #4b5364 ${start}%, #568af2 ${start}%, #568af2 ${start +
            length}%, #4b5364 ${start + length}%, #4b5364 100%)`
        }}
        type="range"
        min="-.5"
        max=".5"
        step="0.00000001"
        value={val}
        className="RangeSlider-range"
        onChange={this.onValueChange}
      />
    )
  }
}

export default RangeSlider
