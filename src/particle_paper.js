import './style.css'
import init, {
  run_particle_paper,
  run_particle_paper_with_cp,
  set_particle_paper_fold_angle,
  set_particle_paper_fold_speed,
  set_particle_paper_hinge_compliance,
  set_particle_paper_hinge_damping,
  set_particle_paper_resolution,
  set_particle_paper_wireframe_enabled,
  set_time_step,
  set_constraint_iters,
  set_num_substeps,
  set_gravity_enabled,
  set_gravity_g,
  set_stretch_enabled,
  set_stretch_compliance,
  set_bending_enabled,
  set_bend_compliance,
  set_self_collision_enabled,
  set_damping,
  set_use_distance_constraints, set_pulling_weight,
} from '../rust/pkg/my_webgpu_app.js'
import { SliderRow, ConstraintGroup, CheckboxRow, Divider, SliderOptions, Button } from './components.js'

function buildPanel() {
  const panel = document.createElement('div')
  panel.className =
    'fixed top-4 left-4 z-10 bg-black/75 backdrop-blur-sm text-white rounded-xl p-4 w-60 text-xs font-mono select-none overflow-y-auto max-h-[calc(100vh-2rem)]'

  const title = document.createElement('div')
  title.className = 'text-sm font-bold mb-3 text-white/90 tracking-wide'
  title.textContent = 'Particle Paper Sim'
  panel.appendChild(title)

  const back = document.createElement('a')
  back.href = '/'
  back.className = 'block mb-3 text-white/40 hover:text-white/80 text-xs'
  back.textContent = '← cloth sim'
  panel.appendChild(back)

  let currentResolution = 32
  panel.appendChild(Button({
    label: 'reset sim',
    onClick: () => set_particle_paper_resolution(Math.round(currentResolution)),
  }))

  panel.appendChild(CheckboxRow({
    label: 'show wireframe',
    checked: false,
    onChange: set_particle_paper_wireframe_enabled,
  }))

  panel.appendChild(SliderRow({
    label: 'resolution',
    min: 1, max: 200, step: 1, value: 32,
    onChange: v => { currentResolution = v; set_particle_paper_resolution(Math.round(v)) },
  }))

  panel.appendChild(Divider())

  // ── Fold control ──
  const foldLabel = document.createElement('div')
  foldLabel.className = 'font-semibold text-white/90 mb-1'
  foldLabel.textContent = 'fold'
  panel.appendChild(foldLabel)

  panel.appendChild(SliderRow({
    label: 'angle (°)',
    min: 0, max: 180, step: 1, value: 0,
    onChange: v => set_particle_paper_fold_angle(v),
  }))

  panel.appendChild(SliderRow({
    label: 'fold speed (°/s)',
    min: 10, max: 720, step: 10, value: 286,
    onChange: v => set_particle_paper_fold_speed(v * Math.PI / 180),
  }))

  panel.appendChild(SliderRow({
    label: 'compliance (1e-x)',
    min: 2, max: 6, step: 0.1, value: 4,
    onChange: v => set_particle_paper_hinge_compliance(Math.pow(10, -v)),
  }))

  panel.appendChild(SliderRow({
    label: 'hinge damping',
    min: 0, max: 5, step: 0.1, value: 0.5,
    onChange: set_particle_paper_hinge_damping,
  }))

  panel.appendChild(Divider())

  // ── Physics ──
  panel.appendChild(SliderRow({
    label: 'damping', min: 0, max: 0.5, step: 0.01, value: 0.0,
    onChange: set_damping,
  }))

  panel.appendChild(SliderRow({
    label: 'time step', min: 0.0001, max: 0.05, step: 0.0001, value: 0.005,
    onChange: set_time_step,
  }))

  panel.appendChild(SliderRow({
    label: 'iterations', min: 1, max: 30, step: 1, value: 1,
    onChange: v => set_constraint_iters(Math.round(v)),
  }))

  panel.appendChild(SliderRow({
    label: 'substeps', min: 1, max: 80, step: 1, value: 50,
    onChange: v => set_num_substeps(Math.round(v)),
  }))

  panel.appendChild(Divider())

  panel.appendChild(ConstraintGroup({
    label: 'gravity',
    enabled: false,
    onToggle: set_gravity_enabled,
    sliders: [
      new SliderOptions({
        weight: -9.8, weightLabel: 'g', weightMin: -20, weightMax: 0, weightStep: 0.1,
        onWeightChange: set_gravity_g,
      }),
    ],
  }))

  panel.appendChild(Divider())

  panel.appendChild(ConstraintGroup({
    label: 'stretch',
    enabled: true,
    onToggle: set_stretch_enabled,
    sliders: [
      new SliderOptions({
        weight: 10,
        weightLabel: 'compliance (1e-x)',
        weightMin: 2, weightMax: 12, weightStep: 0.1,
        onWeightChange: v => set_stretch_compliance(Math.pow(10, -v)),
      }),
    ],
  }))

  panel.appendChild(ConstraintGroup({
    label: 'bending',
    enabled: true,
    onToggle: set_bending_enabled,
    sliders: [
      new SliderOptions({
        weight: 10,
        weightLabel: 'compliance (1e-x)',
        weightMin: 2, weightMax: 12, weightStep: 0.1,
        onWeightChange: v => set_bend_compliance(Math.pow(10, -v)),
      }),
    ],
  }))

  panel.appendChild(Divider())

  panel.appendChild(CheckboxRow({
    label: 'self-collision',
    checked: false,
    onChange: set_self_collision_enabled,
  }))

  document.body.appendChild(panel)
}

init().then(async () => {
  // Particle paper defaults: XPBD, no gravity, stretch+bend stiff, hinges drive folds.
  set_use_distance_constraints(true)
  set_gravity_enabled(false)
  set_self_collision_enabled(true)
  set_stretch_enabled(true)
  set_bending_enabled(true)
  set_particle_paper_resolution(64)
  set_stretch_compliance(1e-10)
  set_bend_compliance(1e-10)
  set_constraint_iters(1)
  set_num_substeps(20)
  set_pulling_weight(0.1)
  set_time_step(0.0005)
  set_damping(0.005)

  try {
    const file = 'assets/waterbomb_tess.cp'
    const cpResponse = await fetch(file)
    if (cpResponse.ok) {
      const cpData = await cpResponse.text()
      await run_particle_paper_with_cp('canvas', cpData)
      console.log(`Loaded crease pattern: ${file}`)
    } else {
      console.log('No crease pattern found, using simple particle paper sim')
      await run_particle_paper('canvas')
    }
  } catch (e) {
    console.error('run_particle_paper() failed:', e)
  }

  // Apply paper-flavored defaults *after* run starts (hinges only exist post-build).
  set_particle_paper_hinge_compliance(1e-6)

  buildPanel()
})
