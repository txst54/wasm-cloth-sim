import './style.css'
import init, {
  run_particle_cloth,
  set_particle_resolution,
  set_particle_sphere,
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
import { SliderRow, ConstraintGroup, CheckboxRow, Divider, SliderOptions, Button } from './components.js'

function buildPanel() {
  const panel = document.createElement('div')
  panel.className =
    'fixed top-4 left-4 z-10 bg-black/75 backdrop-blur-sm text-white rounded-xl p-4 w-60 text-xs font-mono select-none overflow-y-auto max-h-[calc(100vh-2rem)]'

  const title = document.createElement('div')
  title.className = 'text-sm font-bold mb-3 text-white/90 tracking-wide'
  title.textContent = 'Particle Cloth + SDF'
  panel.appendChild(title)

  const back = document.createElement('a')
  back.href = '/'
  back.className = 'block mb-3 text-white/40 hover:text-white/80 text-xs'
  back.textContent = '← cloth sim'
  panel.appendChild(back)

  let currentResolution = 32
  panel.appendChild(Button({
    label: 'reset sim',
    onClick: () => set_particle_resolution(Math.round(currentResolution)),
  }))

  panel.appendChild(SliderRow({
    label: 'resolution', min: 4, max: 200, step: 1, value: 32,
    onChange: v => { currentResolution = v; set_particle_resolution(Math.round(v)) },
  }))

  panel.appendChild(Divider())

  // ── Substepping / integrator ───────────────────────────────────────────────
  panel.appendChild(SliderRow({
    label: 'time step', min: 0.0005, max: 0.05, step: 0.0005, value: 0.01,
    onChange: set_time_step,
  }))
  panel.appendChild(SliderRow({
    label: 'substeps', min: 1, max: 80, step: 1, value: 20,
    onChange: v => set_num_substeps(Math.round(v)),
  }))
  panel.appendChild(SliderRow({
    label: 'damping', min: 0, max: 0.5, step: 0.005, value: 0.0,
    onChange: set_damping,
  }))

  panel.appendChild(Divider())

  // ── Gravity ────────────────────────────────────────────────────────────────
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

  // ── Stretch (distance constraint, compliance log scale) ────────────────────
  panel.appendChild(ConstraintGroup({
    label: 'stretch',
    enabled: true,
    onToggle: set_stretch_enabled,
    sliders: [
      new SliderOptions({
        weight: 7,
        weightLabel: 'compliance (1e-x)',
        weightMin: 2, weightMax: 10, weightStep: 0.1,
        onWeightChange: v => set_stretch_compliance(Math.pow(10, -v)),
      }),
    ],
  }))

  // ── Bending (distance constraint on diamond diagonal) ──────────────────────
  panel.appendChild(ConstraintGroup({
    label: 'bending',
    enabled: true,
    onToggle: set_bending_enabled,
    sliders: [
      new SliderOptions({
        weight: 6,
        weightLabel: 'compliance (1e-x)',
        weightMin: 2, weightMax: 10, weightStep: 0.1,
        onWeightChange: v => set_bend_compliance(Math.pow(10, -v)),
      }),
    ],
  }))

  panel.appendChild(Divider())

  // ── Self-collision (particle-vs-particle hash) ─────────────────────────────
  panel.appendChild(ConstraintGroup({
    label: 'self-collision (hash)',
    enabled: true,
    onToggle: set_self_collision_enabled,
    sliders: [
      new SliderOptions({
        weight: 0.45,
        weightLabel: 'particle radius / edge',
        weightMin: 0.1, weightMax: 0.6, weightStep: 0.01,
        onWeightChange: set_particle_radius_scale,
      }),
    ],
  }))

  panel.appendChild(Divider())

  // ── SDF rigid sphere obstacle ──────────────────────────────────────────────
  const sphereLabel = document.createElement('div')
  sphereLabel.className = 'font-semibold text-white/90 mb-1'
  sphereLabel.textContent = 'SDF sphere'
  panel.appendChild(sphereLabel)

  const sphere = { cx: 0.0, cy: 0.0, cz: 0.0, r: 0.4 }
  const push = () => set_particle_sphere(sphere.cx, sphere.cy, sphere.cz, sphere.r)

  panel.appendChild(SliderRow({
    label: 'cx', min: -1.0, max: 1.0, step: 0.01, value: sphere.cx,
    onChange: v => { sphere.cx = v; push() },
  }))
  panel.appendChild(SliderRow({
    label: 'cy', min: -1.5, max: 1.0, step: 0.01, value: sphere.cy,
    onChange: v => { sphere.cy = v; push() },
  }))
  panel.appendChild(SliderRow({
    label: 'cz', min: -1.0, max: 1.0, step: 0.01, value: sphere.cz,
    onChange: v => { sphere.cz = v; push() },
  }))
  panel.appendChild(SliderRow({
    label: 'radius', min: 0.05, max: 0.9, step: 0.01, value: sphere.r,
    onChange: v => { sphere.r = v; push() },
  }))

  panel.appendChild(Divider())
  panel.appendChild(CheckboxRow({
    label: 'pin (mouse drag)', checked: true, onChange: set_pin_enabled,
  }))

  document.body.appendChild(panel)
}

init().then(async () => {
  // Particle-cloth defaults: distance constraints only, lots of substeps, hard contacts.
  set_use_distance_constraints(true)
  set_constraint_iters(1)
  set_num_substeps(50)
  set_time_step(0.005)
  set_damping(0.0001)
  set_gravity_enabled(true)
  set_gravity_g(-9.8)
  set_stretch_enabled(true)
  set_bending_enabled(true)
  set_stretch_compliance(1e-10)
  set_bend_compliance(1e-10)
  set_self_collision_enabled(true)
  set_pin_enabled(true)

  try {
    await run_particle_cloth('canvas')
  } catch (e) {
    console.error('run_particle_cloth() failed:', e)
  }
  buildPanel()
})
