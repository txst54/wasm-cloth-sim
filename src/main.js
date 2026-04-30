import './style.css'
import init, {
  run,
  set_time_step,
  set_constraint_iters,
  set_num_substeps,
  set_gravity_enabled,
  set_gravity_g,
  set_pin_enabled,
  set_pin_weight,
  set_stretch_enabled,
  set_stretch_weight,
  set_stretch_compliance,
  set_bending_enabled,
  set_bending_weight,
  set_bend_compliance,
  set_pulling_enabled,
  set_pulling_weight,
  set_self_collision_enabled,
  set_self_collision_threshold,
  set_self_collision_recompute_iters,
  set_use_distance_constraints,
  set_resolution,
  set_pulling_area,
} from '../rust/pkg/my_webgpu_app.js'
import {
  CheckboxRow, Divider, Button,
  Param, Toggle, ParamConstraintGroup, NavPanel,
} from './components.js'

const USE_DC_DEFAULT = false

// ── Single source of truth for params ─────────────────────────────────────
const P = {
  timeStep:   new Param({ label: 'time step',  min: 0.001, max: 0.1, step: 0.001, value: 0.01,  onChange: set_time_step }),
  iters:      new Param({ label: 'iterations', min: 1, max: 30, step: 1, value: 5, asInt: true, onChange: set_constraint_iters }),
  substeps:   new Param({ label: 'substeps',   min: 1, max: 40, step: 1, value: 10, asInt: true, onChange: set_num_substeps }),
  resolution: new Param({ label: 'resolution', min: 4, max: 128, step: 1, value: 32, asInt: true, onChange: set_resolution }),

  gravityG:   new Param({ label: 'Gravity G',  min: -20, max: 0, step: 0.1, value: -9.8, onChange: set_gravity_g }),
  pinWeight:  new Param({ label: 'Pin Weight', min: 0,   max: 1, step: 0.05, value: 1.0, onChange: set_pin_weight }),

  stretchWeight:     new Param({ label: 'Stretching Weight', min: 0, max: 1, step: 0.05, value: 0.5, onChange: set_stretch_weight }),
  stretchCompliance: new Param({ label: 'Compliance (1e-x)', min: 2, max: 10, step: 0.1, value: 7,
                                 onChange: set_stretch_compliance, transform: v => Math.pow(10, -v) }),
  bendingWeight:     new Param({ label: 'Bending Weight',    min: 0, max: 1, step: 0.05, value: 0.5, onChange: set_bending_weight }),
  bendCompliance:    new Param({ label: 'Compliance (1e-x)', min: 2, max: 10, step: 0.1, value: 6,
                                 onChange: set_bend_compliance, transform: v => Math.pow(10, -v) }),

  pullingWeight: new Param({ label: 'Pulling Weight', min: 0, max: 0.5, step: 0.02, value: 0.1, onChange: set_pulling_weight }),
  pullingArea:   new Param({ label: 'Pulling Area',   min: 0, max: 20,  step: 1,    value: 5,   onChange: set_pulling_area }),

  selfCollThresh: new Param({ label: 'Collision Threshold', min: 0.001, max: 0.1, step: 0.001, value: 0.01, onChange: set_self_collision_threshold }),
  selfCollIters:  new Param({ label: 'Recompute Iters',     min: 1, max: 5, step: 1, value: 1, asInt: true, onChange: set_self_collision_recompute_iters }),
}

const T = {
  gravity:       new Toggle({ enabled: true, onChange: set_gravity_enabled }),
  pin:           new Toggle({ enabled: true, onChange: set_pin_enabled }),
  stretch:       new Toggle({ enabled: true, onChange: set_stretch_enabled }),
  bending:       new Toggle({ enabled: true, onChange: set_bending_enabled }),
  pulling:       new Toggle({ enabled: true, onChange: set_pulling_enabled }),
  selfCollision: new Toggle({ enabled: true, onChange: set_self_collision_enabled }),
}

function applyAllInitParams() {
  Object.values(P).forEach(p => p.apply())
  Object.values(T).forEach(t => t.apply())
  set_use_distance_constraints(USE_DC_DEFAULT)
}

function buildPanel() {
  const panel = document.createElement('div')
  panel.className = 'fixed top-4 left-4 z-10 bg-black/75 backdrop-blur-sm text-white rounded-xl p-4 w-60 text-xs font-mono select-none overflow-y-auto max-h-[calc(100vh-2rem)]'

  const title = document.createElement('div')
  title.className = 'text-sm font-bold mb-3 text-white/90 tracking-wide'
  title.textContent = 'Sim Params'
  panel.appendChild(title)

  panel.appendChild(Button({ label: 'reset sim', onClick: () => P.resolution.apply() }))

  panel.appendChild(P.timeStep.slider())
  panel.appendChild(P.iters.slider())
  const substepRow = P.substeps.slider()
  panel.appendChild(substepRow)

  panel.appendChild(Divider())

  panel.appendChild(ParamConstraintGroup({ label: 'gravity', toggle: T.gravity, params: [P.gravityG] }))

  panel.appendChild(Divider())

  panel.appendChild(CheckboxRow({
    label: 'distance constraints',
    checked: USE_DC_DEFAULT,
    indent: false,
    onChange: v => { set_use_distance_constraints(v); applyDistanceConstraintsMode(v) },
  }))

  panel.appendChild(Divider())

  panel.appendChild(ParamConstraintGroup({ label: 'pin', toggle: T.pin, params: [P.pinWeight] }))

  const stretchStiffness  = ParamConstraintGroup({ label: 'stretch', toggle: T.stretch, params: [P.stretchWeight] })
  const stretchCompliance = ParamConstraintGroup({ label: 'stretch', toggle: T.stretch, params: [P.stretchCompliance] })
  panel.appendChild(stretchStiffness)
  panel.appendChild(stretchCompliance)

  const bendStiffness  = ParamConstraintGroup({ label: 'bending', toggle: T.bending, params: [P.bendingWeight] })
  const bendCompliance = ParamConstraintGroup({ label: 'bending', toggle: T.bending, params: [P.bendCompliance] })
  panel.appendChild(bendStiffness)
  panel.appendChild(bendCompliance)

  panel.appendChild(ParamConstraintGroup({
    label: 'pulling', toggle: T.pulling, params: [P.pullingWeight, P.pullingArea],
  }))

  panel.appendChild(ParamConstraintGroup({
    label: 'self collision', toggle: T.selfCollision, params: [P.selfCollThresh, P.selfCollIters],
  }))

  panel.appendChild(Divider())
  panel.appendChild(P.resolution.slider())

  document.body.appendChild(panel)

  function applyDistanceConstraintsMode(on) {
    stretchStiffness.style.display  = on ? 'none' : ''
    bendStiffness.style.display     = on ? 'none' : ''
    stretchCompliance.style.display = on ? '' : 'none'
    bendCompliance.style.display    = on ? '' : 'none'
    substepRow.style.display        = on ? '' : 'none'
  }
  applyDistanceConstraintsMode(USE_DC_DEFAULT)
}

init().then(async () => {
  applyAllInitParams()

  try {
    await run('canvas')
  } catch (e) {
    console.error('WASM run() failed:', e)
  }
  buildPanel()
  NavPanel()
})
