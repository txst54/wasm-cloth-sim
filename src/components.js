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
  onToggle,
  sliders,
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
  if (sliders) {
    for (const s of sliders) {
      if (s.weight !== undefined && s.onWeightChange) {
        root.appendChild(SliderRow({
          label: s.weightLabel,
          min: s.weightMin,
          max: s.weightMax,
          step: s.weightStep,
          value: s.weight,
          indent: true,
          onChange: s.onWeightChange,
        }))
      }
    }
  }
  return root
}

/**
 * CheckboxRow — a single labeled checkbox, optionally indented.
 *
 * Props:
 *   label    {string}
 *   checked  {boolean}  initial state
 *   indent   {boolean}  left-indent (for sub-params, default true)
 *   onChange {fn}       called with boolean on change
 *
 * Returns a <div> DOM node.
 */
export function CheckboxRow({ label, checked, indent = true, onChange }) {
  const root = el('div', 'mb-1')
  const checkLabel = el('label', `flex items-center gap-2 cursor-pointer${indent ? ' pl-4' : ''}`)
  const checkbox = el('input', 'accent-white cursor-pointer', { type: 'checkbox' })
  checkbox.checked = checked
  const checkText = el('span', 'text-white/60')
  checkText.textContent = label
  checkLabel.append(checkbox, checkText)
  checkbox.addEventListener('change', () => onChange(checkbox.checked))
  root.appendChild(checkLabel)
  return root
}

/**
 * Divider — a thin horizontal rule.
 */
export function Divider() {
  return el('div', 'border-t border-white/10 my-2')
}

/**
 * Button — a full-width clickable button.
 *
 * Props:
 *   label    {string}
 *   onClick  {fn}
 */
export function Button({ label, onClick }) {
  const btn = el('button', 'w-full mb-2 px-2 py-1 bg-white/10 hover:bg-white/20 active:bg-white/30 border border-white/20 rounded text-white/90 cursor-pointer transition-colors')
  btn.type = 'button'
  btn.textContent = label
  btn.addEventListener('click', onClick)
  return btn
}

/**
 * Param — single source of truth for a numeric parameter.
 * Drives both the WASM setter on init (.apply()) and the slider UI (.slider()).
 * Mutating .value (via slider) keeps subsequent .apply() calls in sync.
 */
export class Param {
  constructor({ label, min, max, step, value, onChange, transform = v => v, asInt = false }) {
    Object.assign(this, { label, min, max, step, value, onChange, transform, asInt })
  }
  get applied() {
    const v = this.asInt ? Math.round(this.value) : this.value
    return this.transform(v)
  }
  apply() { this.onChange(this.applied) }
  slider({ indent = false } = {}) {
    return SliderRow({
      label: this.label, min: this.min, max: this.max, step: this.step, value: this.value, indent,
      onChange: v => { this.value = v; this.apply() },
    })
  }
  toSliderOptions() {
    return new SliderOptions({
      weight: this.value,
      weightLabel: this.label,
      weightMin: this.min,
      weightMax: this.max,
      weightStep: this.step,
      onWeightChange: v => { this.value = v; this.apply() },
    })
  }
}

/**
 * Toggle — single source of truth for a boolean parameter.
 */
export class Toggle {
  constructor({ enabled, onChange }) {
    this.enabled = enabled
    this.onChange = onChange
  }
  apply() { this.onChange(this.enabled) }
}

/** Build a ConstraintGroup driven by a Toggle + array of Param. */
export function ParamConstraintGroup({ label, toggle, params }) {
  return ConstraintGroup({
    label,
    enabled: toggle.enabled,
    onToggle: v => { toggle.enabled = v; toggle.onChange(v) },
    sliders: params.map(p => p.toSliderOptions()),
  })
}

export const SIMS = [
  { path: '/',                label: 'cloth' },
  { path: '/paper',           label: 'paper' },
  { path: '/particle',        label: 'particle cloth' },
  { path: '/particle_paper',  label: 'particle paper' },
]

/** NavPanel — fixed top-right links to the other simulations. */
export function NavPanel() {
  const here = location.pathname
  const isCurrent = (p) => {
    if (p === '/')               return here === '/' || (!here.startsWith('/paper') && !here.startsWith('/particle'))
    if (p === '/particle_paper') return here.startsWith('/particle_paper')
    if (p === '/particle')       return here.startsWith('/particle') && !here.startsWith('/particle_paper')
    if (p === '/paper')          return here.startsWith('/paper')
    return false
  }

  const panel = el('div', 'fixed top-4 right-4 z-10 bg-black/75 backdrop-blur-sm text-white rounded-xl p-3 text-xs font-mono select-none')
  const title = el('div', 'text-[10px] uppercase tracking-wider text-white/50 mb-2')
  title.textContent = 'simulations'
  panel.appendChild(title)

  for (const s of SIMS) {
    const a = el('a', 'block py-0.5 ' + (isCurrent(s.path)
      ? 'text-white/90 font-semibold'
      : 'text-white/50 hover:text-white/90'))
    a.href = s.path
    a.textContent = s.label
    panel.appendChild(a)
  }
  document.body.appendChild(panel)
  return panel
}

export const CREASE_PATTERNS = [
  'line.cp',
  'lines.cp',
  'miura_ori.cp',
  'waterbomb.cp',
  'waterbomb_tess.cp',
  'hex_torso.cp'
]

/**
 * CreasePatternDropdown — labeled <select> of crease patterns.
 *
 * Props:
 *   initial  {string}  filename (e.g. 'miura_ori.cp')
 *   onLoad   {fn}      called with cpData string after fetch succeeds
 */
export function CreasePatternDropdown({ initial, onLoad }) {
  const root = el('div', 'mb-2')

  const labelRow = el('div', 'flex justify-between items-center mb-1')
  const labelEl = el('span', 'text-white/60')
  labelEl.textContent = 'crease pattern'
  labelRow.appendChild(labelEl)

  const sel = el('select', 'w-full bg-black/60 border border-white/20 rounded px-1 py-0.5 text-white/90 cursor-pointer focus:outline-none focus:border-sky-400')
  for (const f of CREASE_PATTERNS) {
    const opt = document.createElement('option')
    opt.value = f
    opt.textContent = f.replace(/\.cp$/, '')
    if (f === initial) opt.selected = true
    sel.appendChild(opt)
  }
  sel.addEventListener('change', async () => {
    const file = `assets/${sel.value}`
    try {
      const r = await fetch(file)
      if (!r.ok) throw new Error(`fetch ${file} -> ${r.status}`)
      const cpData = await r.text()
      onLoad(cpData)
    } catch (e) {
      console.error('crease pattern load failed:', e)
    }
  })

  root.append(labelRow, sel)
  return root
}

export class SliderOptions {
  constructor({
                weight,
                weightLabel = 'weight',
                weightMin = 0,
                weightMax = 1,
                weightStep = 0.05,
                onWeightChange,
              } = {}) {
    this.weight = weight;
    this.weightLabel = weightLabel;
    this.weightMin = weightMin;
    this.weightMax = weightMax;
    this.weightStep = weightStep;
    this.onWeightChange = onWeightChange;
  }
}