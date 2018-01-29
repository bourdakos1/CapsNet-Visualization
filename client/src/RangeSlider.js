import React, { Component } from 'react'
import './RangeSlider.css'

class RangeSlider extends Component {
  render() {
    return (
      <div className="RangeSlider-slider">
        <div
          style={{
            left: `${this.props.value * 100 + 50}%`,
            transform: `translate(0, calc(-50% + 1px)) scaleX(calc(${this.props
              .default + 0.5} - ${this.props.value + 0.5}))`
          }}
          className="RangeSlider-fill"
        />
        <div
          style={{ left: `${this.props.default * 100 + 50}%` }}
          className="RangeSlider-lock"
        />
        <div
          style={{ left: `${this.props.value * 100 + 50}%` }}
          className="RangeSlider-nob"
        />
      </div>
    )
  }
}

export default RangeSlider
