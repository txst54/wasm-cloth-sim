import './style.css'
import init, {
  run_head_cloth,
  set_particle_resolution,
  set_particle_radius_scale,
  set_time_step,
  set_num_substeps,
  set_constraint_iters,
  set_gravity_enabled,
  set_gravity_g,
  set_pin_enabled,
  set_stretch_enabled,
  set_stretch_compliance,
  set_bending_enabled,
  set_bend_compliance,
  set_self_collision_enabled,
  set_use_distance_constraints,
  set_damping,
} from '../rust/pkg/my_webgpu_app.js'
import {
  CheckboxRow, Divider, Button,
  Param, Toggle, ParamConstraintGroup, NavPanel,
} from './components.js'

const USE_DC_DEFAULT = true

const P = {
  resolution: new Param({ label: 'resolution', min: 4, max: 200, step: 1, value: 120, asInt: true, onChange: set_particle_resolution }),

  timeStep: new Param({ label: 'time step', min: 0.0005, max: 0.05, step: 0.0005, value: 0.005, onChange: set_time_step }),
  substeps: new Param({ label: 'substeps',  min: 1, max: 80, step: 1, value: 5, asInt: true, onChange: set_num_substeps }),
  iters:    new Param({ label: 'iterations', min: 1, max: 30, step: 1, value: 1, asInt: true, onChange: set_constraint_iters }),
  damping:  new Param({ label: 'damping',   min: 0, max: 0.5, step: 0.0005, value: 0.0005, onChange: set_damping }),

  gravityG: new Param({ label: 'g', min: -20, max: 0, step: 0.1, value: -9.8, onChange: set_gravity_g }),

  stretchCompliance: new Param({ label: 'compliance (1e-x)', min: 2, max: 12, step: 0.1, value: 10,
                                 onChange: set_stretch_compliance, transform: v => Math.pow(10, -v) }),
  bendCompliance:    new Param({ label: 'compliance (1e-x)', min: 2, max: 12, step: 0.1, value: 10,
                                 onChange: set_bend_compliance, transform: v => Math.pow(10, -v) }),

  particleRadius: new Param({ label: 'particle radius / edge', min: 0.1, max: 0.6, step: 0.01, value: 0.45, onChange: set_particle_radius_scale }),
}

const T = {
  gravity:       new Toggle({ enabled: true, onChange: set_gravity_enabled }),
  stretch:       new Toggle({ enabled: true, onChange: set_stretch_enabled }),
  bending:       new Toggle({ enabled: true, onChange: set_bending_enabled }),
  selfCollision: new Toggle({ enabled: true, onChange: set_self_collision_enabled }),
  pin:           new Toggle({ enabled: false, onChange: set_pin_enabled }),
}

function applyAllInitParams() {
  set_use_distance_constraints(USE_DC_DEFAULT)
  Object.values(P).forEach(p => p.apply())
  Object.values(T).forEach(t => t.apply())
}

function buildPanel() {
  const panel = document.createElement('div')
  panel.className =
    'fixed top-4 left-4 z-10 bg-black/75 backdrop-blur-sm text-white rounded-xl p-4 w-60 text-xs font-mono select-none overflow-y-auto max-h-[calc(100vh-2rem)]'

  const title = document.createElement('div')
  title.className = 'text-sm font-bold mb-3 text-white/90 tracking-wide'
  title.textContent = 'Cloth on Head'
  panel.appendChild(title)

  panel.appendChild(Button({ label: 'reset sim', onClick: () => P.resolution.apply() }))

  panel.appendChild(P.resolution.slider())
  panel.appendChild(Divider())
  panel.appendChild(P.timeStep.slider())
  panel.appendChild(P.substeps.slider())
  panel.appendChild(P.damping.slider())
  panel.appendChild(Divider())
  panel.appendChild(ParamConstraintGroup({ label: 'gravity', toggle: T.gravity, params: [P.gravityG] }))
  panel.appendChild(Divider())
  panel.appendChild(ParamConstraintGroup({ label: 'stretch', toggle: T.stretch, params: [P.stretchCompliance] }))
  panel.appendChild(ParamConstraintGroup({ label: 'bending', toggle: T.bending, params: [P.bendCompliance] }))
  panel.appendChild(Divider())
  panel.appendChild(ParamConstraintGroup({
    label: 'self-collision (hash)', toggle: T.selfCollision, params: [P.particleRadius],
  }))
  panel.appendChild(Divider())
  panel.appendChild(CheckboxRow({
    label: 'pin (mouse drag)', checked: T.pin.enabled,
    onChange: v => { T.pin.enabled = v; T.pin.onChange(v) },
  }))

  document.body.appendChild(panel)
}

async function loadObj(url) {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`failed to fetch ${url}: ${r.status}`)
  return await r.text()
}

init().then(async () => {
  applyAllInitParams()

  let objText
  try {
    objText = await loadObj('/assets/male_head1.obj')
  } catch (e) {
    console.error('failed to load head OBJ:', e)
    return
  }

  buildPanel()
  NavPanel()
  try {
    await run_head_cloth('canvas', objText)
  } catch (e) {
    console.error('run_head_cloth() failed:', e)
  }
})
