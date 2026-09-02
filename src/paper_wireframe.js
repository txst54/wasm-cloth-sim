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
  set_surface_enabled,
  set_use_distance_constraints,
  set_wireframe_regular_color,
  set_wireframe_mountain_color,
  set_wireframe_valley_color,
  set_paper_spin_speed,
  drain_paper_trace,
} from '../rust/pkg/my_webgpu_app.js'
import {
  CheckboxRow, Divider, Button,
  Param, Toggle, ParamConstraintGroup, NavPanel, CreasePatternDropdown, LoadingOverlay,
} from './components.js'
import { startSimConsole } from './sim_console.js'

const USE_DC_DEFAULT = false
const INITIAL_CP = 'hex_torso.cp'

// Constant camera auto-rotation about a vertical axis through the model
// centre, in degrees per second. Set to 0 to disable.
const SPIN_SPEED_DEG_PER_SEC = 10

// Wireframe edge colors, RGB in 0..1. Edit these to recolor the wireframe.
//   regular  – non-crease mesh edges
//   mountain – mountain-fold creases
//   valley   – valley-fold creases
const WIREFRAME_COLORS = {
  regular:  [0.25, 0.25, 0.25],
  mountain: [0.7, 0.7, 0.7],
  valley:   [0.501, 0.5127, 0.23578],
}

function applyWireframeColors(c) {
  set_wireframe_regular_color(...c.regular)
  set_wireframe_mountain_color(...c.mountain)
  set_wireframe_valley_color(...c.valley)
}

const P = {
  resolution: new Param({ label: 'resolution', min: 1, max: 64, step: 1, value: 1, asInt: true, onChange: set_paper_resolution,
                          overlayLabel: v => `Triangulating (resolution ${v})…` }),

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

const post_P = {
  foldAngle:  new Param({ label: 'angle (°)',       min: 0, max: 180, step: 1, value: 90, onChange: set_paper_fold_angle })
}
const post_T = {
  wireframe:     new Toggle({ enabled: true, onChange: set_wireframe_enabled }),
  surface:       new Toggle({ enabled: false, onChange: set_surface_enabled })
}

function applyAllInitParams(params, toggles) {
  Object.values(params).forEach(p => p.apply())
  Object.values(toggles).forEach(t => t.apply())
  set_use_distance_constraints(USE_DC_DEFAULT)
}

init().then(async () => {
  applyAllInitParams(P, T)
  startSimConsole(drain_paper_trace)

  await LoadingOverlay.withLoading('Loading crease pattern…', async () => {
    try {
      const file = `../assets/${INITIAL_CP}`
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
  })
  applyAllInitParams(post_P, post_T)
  applyWireframeColors(WIREFRAME_COLORS)
  set_paper_spin_speed(SPIN_SPEED_DEG_PER_SEC)
})
