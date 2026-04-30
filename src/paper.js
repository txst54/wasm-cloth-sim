import './style.css'
import init, {
  run_paper,
  run_paper_with_cp,
  set_paper_fold_angle,
  set_paper_fold_speed,
  set_paper_hinge_compliance,
  set_paper_hinge_damping,
  set_paper_resolution,
  set_time_step,
  set_constraint_iters,
  set_num_substeps,
  set_gravity_enabled,
  set_gravity_g,
  set_stretch_enabled,
  set_stretch_weight,
  set_stretch_compliance,
  set_bending_enabled,
  set_bending_weight,
  set_bend_compliance,
  set_pulling_enabled,
  set_pulling_weight,
  set_pulling_area,
  set_self_collision_enabled,
  set_damping,
  set_wireframe_enabled,
  set_use_distance_constraints,
} from '../rust/pkg/my_webgpu_app.js'
import {
  CheckboxRow, Divider, Button,
  Param, Toggle, ParamConstraintGroup, NavPanel, CreasePatternDropdown,
} from './components.js'

const USE_DC_DEFAULT = false
const INITIAL_CP = 'miura_ori.cp'

const P = {
  resolution: new Param({ label: 'resolution', min: 1, max: 64, step: 1, value: 1, asInt: true, onChange: set_paper_resolution }),

  foldAngle:  new Param({ label: 'angle (°)',       min: 0, max: 180, step: 1, value: 0, onChange: set_paper_fold_angle }),
  foldSpeed:  new Param({ label: 'fold speed (°/s)', min: 10, max: 720, step: 10, value: 286,
                          onChange: set_paper_fold_speed, transform: v => v * Math.PI / 180 }),
  hingeCompliance: new Param({ label: 'compliance (1e-x)', min: 2, max: 10, step: 0.1, value: 6,
                               onChange: set_paper_hinge_compliance, transform: v => Math.pow(10, -v) }),
  hingeDamping: new Param({ label: 'hinge damping', min: 0, max: 5, step: 0.1, value: 0.5, onChange: set_paper_hinge_damping }),

  damping:  new Param({ label: 'damping',   min: 0, max: 1, step: 0.01, value: 0.7, onChange: set_damping }),
  timeStep: new Param({ label: 'time step', min: 0.00001, max: 0.1, step: 0.0001, value: 0.001, onChange: set_time_step }),
  iters:    new Param({ label: 'iterations', min: 1, max: 30, step: 1, value: 15, asInt: true, onChange: set_constraint_iters }),
  substeps: new Param({ label: 'substeps',   min: 1, max: 100, step: 1, value: 50, asInt: true, onChange: set_num_substeps }),

  gravityG: new Param({ label: 'g', min: -20, max: 0, step: 0.1, value: -9.8, onChange: set_gravity_g }),

  stretchWeight:     new Param({ label: 'weight', min: 0, max: 1, step: 0.05, value: 1.0, onChange: set_stretch_weight }),
  stretchCompliance: new Param({ label: 'Compliance (1e-x)', min: 2, max: 12, step: 0.1, value: 10,
                                 onChange: set_stretch_compliance, transform: v => Math.pow(10, -v) }),
  bendingWeight:     new Param({ label: 'weight', min: 0, max: 1, step: 0.05, value: 1.0, onChange: set_bending_weight }),
  bendCompliance:    new Param({ label: 'Compliance (1e-x)', min: 2, max: 12, step: 0.1, value: 10,
                                 onChange: set_bend_compliance, transform: v => Math.pow(10, -v) }),

  pullingWeight: new Param({ label: 'weight', min: 0, max: 0.5, step: 0.02, value: 0.1, onChange: set_pulling_weight }),
  pullingArea:   new Param({ label: 'area',   min: 0, max: 20,  step: 1,    value: 5,   onChange: set_pulling_area }),
}

const T = {
  gravity:       new Toggle({ enabled: false, onChange: set_gravity_enabled }),
  stretch:       new Toggle({ enabled: true,  onChange: set_stretch_enabled }),
  bending:       new Toggle({ enabled: true,  onChange: set_bending_enabled }),
  pulling:       new Toggle({ enabled: true,  onChange: set_pulling_enabled }),
  selfCollision: new Toggle({ enabled: false, onChange: set_self_collision_enabled }),
}

function applyAllInitParams() {
  Object.values(P).forEach(p => p.apply())
  Object.values(T).forEach(t => t.apply())
  set_use_distance_constraints(USE_DC_DEFAULT)
}

function buildPanel() {
  const panel = document.createElement('div')
  panel.className =
    'fixed top-4 left-4 z-10 bg-black/75 backdrop-blur-sm text-white rounded-xl p-4 w-60 text-xs font-mono select-none overflow-y-auto max-h-[calc(100vh-2rem)]'

  const title = document.createElement('div')
  title.className = 'text-sm font-bold mb-3 text-white/90 tracking-wide'
  title.textContent = 'Paper Sim'
  panel.appendChild(title)

  panel.appendChild(Button({ label: 'reset sim', onClick: () => P.resolution.apply() }))

  panel.appendChild(CreasePatternDropdown({
    initial: INITIAL_CP,
    onLoad: async cpData => {
      try {
        await run_paper_with_cp('canvas', cpData)
        applyAllInitParams()
      } catch (e) {
        console.error('run_paper_with_cp() failed:', e)
      }
    },
  }))

  panel.appendChild(Divider())

  panel.appendChild(CheckboxRow({ label: 'show wireframe', checked: false, onChange: set_wireframe_enabled }))
  panel.appendChild(P.resolution.slider())

  panel.appendChild(Divider())

  const foldLabel = document.createElement('div')
  foldLabel.className = 'font-semibold text-white/90 mb-1'
  foldLabel.textContent = 'fold'
  panel.appendChild(foldLabel)

  panel.appendChild(P.foldAngle.slider())
  panel.appendChild(P.foldSpeed.slider())
  panel.appendChild(P.hingeCompliance.slider())
  panel.appendChild(P.hingeDamping.slider())

  panel.appendChild(Divider())

  panel.appendChild(P.damping.slider())
  panel.appendChild(P.timeStep.slider())
  panel.appendChild(P.iters.slider())
  const substepRow = P.substeps.slider()
  panel.appendChild(substepRow)

  panel.appendChild(Divider())

  panel.appendChild(CheckboxRow({
    label: 'distance constraints',
    checked: USE_DC_DEFAULT,
    indent: false,
    onChange: v => { set_use_distance_constraints(v); applyDistanceConstraintsMode(v) },
  }))

  panel.appendChild(Divider())

  panel.appendChild(ParamConstraintGroup({ label: 'gravity', toggle: T.gravity, params: [P.gravityG] }))

  panel.appendChild(Divider())

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
    const file = `assets/${INITIAL_CP}`
    const cpResponse = await fetch(file)
    if (cpResponse.ok) {
      const cpData = await cpResponse.text()
      await run_paper_with_cp('canvas', cpData)
      console.log(`Loaded crease pattern: ${file}`)
    } else {
      console.log('No crease pattern found, using simple paper sim')
      await run_paper('canvas')
    }
  } catch (e) {
    console.error('run_paper() failed:', e)
  }

  buildPanel()
  NavPanel()
})
