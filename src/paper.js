import './style.css'
import init, {
  run_paper,
  run_paper_with_cp,
  set_paper_fold_angle,
  set_paper_fold_speed,
  set_paper_hinge_compliance,
  set_time_step,
  set_constraint_iters,
  set_gravity_enabled,
  set_gravity_g,
  set_stretch_enabled,
  set_stretch_weight,
  set_bending_enabled,
  set_bending_weight,
  set_pulling_enabled,
  set_pulling_weight,
  set_pulling_area,
  set_self_collision_enabled,
  set_damping,
  set_wireframe_enabled,
} from '../rust/pkg/my_webgpu_app.js'
import { SliderRow, ConstraintGroup, CheckboxRow, Divider, SliderOptions } from './components.js'

function buildPanel() {
  const panel = document.createElement('div')
  panel.className =
    'fixed top-4 left-4 z-10 bg-black/75 backdrop-blur-sm text-white rounded-xl p-4 w-60 text-xs font-mono select-none overflow-y-auto max-h-[calc(100vh-2rem)]'

  const title = document.createElement('div')
  title.className = 'text-sm font-bold mb-3 text-white/90 tracking-wide'
  title.textContent = 'Paper Sim'
  panel.appendChild(title)

  // back link
  const back = document.createElement('a')
  back.href = '/'
  back.className = 'block mb-3 text-white/40 hover:text-white/80 text-xs'
  back.textContent = '← cloth sim'
  panel.appendChild(back)

  // Wireframe toggle
  panel.appendChild(CheckboxRow({
    label: 'show wireframe',
    checked: false,
    onChange: set_wireframe_enabled,
  }))

  panel.appendChild(Divider())

  // ── Fold control ──────────────────────────────────────────────────────────
  const foldLabel = document.createElement('div')
  foldLabel.className = 'font-semibold text-white/90 mb-1'
  foldLabel.textContent = 'fold'
  panel.appendChild(foldLabel)

  panel.appendChild(SliderRow({
    label: 'angle (°)',
    min: 0, max: 180, step: 1, value: 0,
    onChange: v => set_paper_fold_angle(v),
  }))

  // fold speed: degrees/s converted to rad/s
  panel.appendChild(SliderRow({
    label: 'fold speed (°/s)',
    min: 10, max: 720, step: 10, value: 286,
    onChange: v => set_paper_fold_speed(v * Math.PI / 180),
  }))

  // compliance on a log scale displayed as exponent
  panel.appendChild(SliderRow({
    label: 'compliance (1e-x)',
    min: 2, max: 6, step: 0.1, value: 4,
    onChange: v => set_paper_hinge_compliance(Math.pow(10, -v)),
  }))

  panel.appendChild(Divider())

  // ── Physics ───────────────────────────────────────────────────────────────
  panel.appendChild(SliderRow({
    label: 'damping',
    min: 0, max: 0.5, step: 0.01, value: 0.7,
    onChange: set_damping,
  }))

  panel.appendChild(SliderRow({
    label: 'time step', min: 0.001, max: 0.1, step: 0.001, value: 0.01,
    onChange: set_time_step,
  }))

  panel.appendChild(SliderRow({
    label: 'iterations', min: 1, max: 30, step: 1, value: 10,
    onChange: v => set_constraint_iters(Math.round(v)),
  }))

  panel.appendChild(Divider())

  panel.appendChild(ConstraintGroup({
    label: 'gravity',
    enabled: true,
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
      new SliderOptions({ weight: 1.0, weightLabel: 'weight', onWeightChange: set_stretch_weight }),
    ],
  }))

  panel.appendChild(ConstraintGroup({
    label: 'bending',
    enabled: true,
    onToggle: set_bending_enabled,
    sliders: [
      new SliderOptions({ weight: 1.0, weightLabel: 'weight', onWeightChange: set_bending_weight }),
    ],
  }))

  panel.appendChild(ConstraintGroup({
    label: 'pulling',
    enabled: true,
    onToggle: set_pulling_enabled,
    sliders: [
      new SliderOptions({
        weight: 0.1, weightStep: 0.02, weightMax: 0.5,
        weightLabel: 'weight', onWeightChange: set_pulling_weight,
      }),
      new SliderOptions({
        weight: 5, weightStep: 1, weightMin: 0, weightMax: 20,
        weightLabel: 'area', onWeightChange: set_pulling_area,
      }),
    ],
  }))

  document.body.appendChild(panel)
}

init().then(async () => {
  // Paper sim defaults
  set_gravity_enabled(true)
  set_self_collision_enabled(false)
  set_bending_enabled(true)
  set_stretch_weight(1.0)
  set_bending_weight(1.0)
  set_constraint_iters(10)
  set_damping(0.7)

  try {
    // Fetch the crease pattern file
    const file = 'assets/waterbomb.cp'
    const cpResponse = await fetch(file)
    if (cpResponse.ok) {
      const cpData = await cpResponse.text()
      await run_paper_with_cp('canvas', cpData)
      console.log(`Loaded crease pattern: ${file}`)
    } else {
      // Fallback to simple paper sim
      console.log('No crease pattern found, using simple paper sim')
      await run_paper('canvas')
    }
  } catch (e) {
    console.error('run_paper() failed:', e)
  }

  buildPanel()
})
