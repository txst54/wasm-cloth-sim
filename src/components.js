/** Create a DOM element with className and optional attributes. */
function el(tag, className, attrs = {}) {
  const node = document.createElement(tag)
  if (className) node.className = className
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v))
  return node
}

/**
 * SliderRow — a labeled range slider paired with a number input.
 *
 * Props:
 *   label    {string}   display label
 *   min, max {number}
 *   step     {number}
 *   value    {number}   initial value
 *   indent   {boolean}  left-indent the label row (for sub-params)
 *   onChange {fn}       called with the new numeric value on any change
 *
 * Returns a <div> DOM node.
 */
export function SliderRow({ label, min, max, step, value, indent = false, onChange }) {
  const root = el('div', 'mb-2')

  const labelRow = el('div', `flex justify-between mb-0.5${indent ? ' pl-4' : ''}`)
  const labelEl  = el('span', 'text-white/60')
  labelEl.textContent = label

  const numInput = el('input', 'w-16 bg-transparent border border-white/20 rounded px-1 text-right text-white/90 focus:outline-none focus:border-sky-400', {
    type: 'number', value, min, max, step,
  })

  const slider = el('input', 'w-full accent-white cursor-pointer', {
    type: 'range', min, max, step, value,
  })

  slider.addEventListener('input', () => {
    numInput.value = slider.value
    onChange(parseFloat(slider.value))
  })

  numInput.addEventListener('change', () => {
    slider.value = numInput.value
    onChange(parseFloat(numInput.value))
  })

  labelRow.append(labelEl, numInput)
  root.append(labelRow, slider)
  return root
}

/**
 * ConstraintGroup — a checkbox toggle for enabled + an optional SliderRow for weight.
 *
 * Props:
 *   label          {string}
 *   enabled        {boolean}  initial checked state
 *   weight         {number}   initial weight value (omit to hide the slider)
 *   weightMin/Max  {number}   defaults 0 / 1
 *   weightStep     {number}   default 0.05
 *   onToggle       {fn}       called with boolean when checkbox changes
 *   onWeightChange {fn}       called with number when slider/input changes
 *
 * Returns a <div> DOM node.
 */
export function ConstraintGroup({
  label,
  enabled,
  weight,
  weightMin = 0,
  weightMax = 1,
  weightStep = 0.05,
  onToggle,
  onWeightChange,
}) {
  const root = el('div', 'mb-2')

  const checkLabel = el('label', 'flex items-center gap-2 mb-1 cursor-pointer')
  const checkbox   = el('input', 'accent-white cursor-pointer', { type: 'checkbox' })
  checkbox.checked = enabled
  const checkText  = el('span', 'font-semibold text-white/90')
  checkText.textContent = label
  checkLabel.append(checkbox, checkText)
  checkbox.addEventListener('change', () => onToggle(checkbox.checked))

  root.appendChild(checkLabel)

  if (weight !== undefined && onWeightChange) {
    root.appendChild(SliderRow({
      label: 'weight',
      min: weightMin,
      max: weightMax,
      step: weightStep,
      value: weight,
      indent: true,
      onChange: onWeightChange,
    }))
  }

  return root
}

/**
 * Divider — a thin horizontal rule.
 */
export function Divider() {
  return el('div', 'border-t border-white/10 my-2')
}
