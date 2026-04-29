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
    run_rigid,
    run_cloth,
} from '../rust/pkg/my_webgpu_app.js'
import {SliderRow, ConstraintGroup, CheckboxRow, Divider, SliderOptions, Button} from './components.js'

const USE_DC_DEFAULT = true

function buildPanel() {
  const panel = document.createElement('div');
  panel.className = 'fixed top-4 left-4 z-10 bg-black/75 backdrop-blur-sm text-white rounded-xl p-4 w-60 text-xs font-mono select-none overflow-y-auto max-h-[calc(100vh-2rem)]';

  const title = document.createElement('div');
  title.className = 'text-sm font-bold mb-3 text-white/90 tracking-wide';
  title.textContent = 'Sim Params';
  panel.appendChild(title);

  let currentResolution = 32;
  panel.appendChild(Button({
    label: 'reset sim',
    onClick: () => set_resolution(Math.round(currentResolution)),
  }));

  panel.appendChild(SliderRow({
    label: 'time step', min: 0.001, max: 0.1, step: 0.001, value: 0.01,
    onChange: set_time_step,
  }));

  panel.appendChild(SliderRow({
    label: 'iterations', min: 1, max: 30, step: 1, value: 5,
    onChange: v => set_constraint_iters(Math.round(v)),
  }));

  // Substeps slider — only meaningful when distance constraints are on.
  const substepRow = SliderRow({
    label: 'substeps', min: 1, max: 40, step: 1, value: 10,
    onChange: v => set_num_substeps(Math.round(v)),
  })
  panel.appendChild(substepRow)

  panel.appendChild(Divider());

  panel.appendChild(ConstraintGroup({
    label: 'gravity',
    enabled: true,
    onToggle: set_gravity_enabled,
    sliders: [
      new SliderOptions({
        weight: -9.8,
        weightLabel: 'Gravity G',
        weightMin: -20,
        weightMax: 0,
        weightStep: 0.1,
        onWeightChange: set_gravity_g,
      }),
    ],
  }));

  panel.appendChild(Divider());

  panel.appendChild(CheckboxRow({
    label: 'distance constraints',
    checked: USE_DC_DEFAULT,
    indent: false,
    onChange: v => {
      set_use_distance_constraints(v)
      applyDistanceConstraintsMode(v)
    },
  }));

  panel.appendChild(Divider());

  panel.appendChild(ConstraintGroup({
    label: 'pin',
    enabled: true,
    onToggle: set_pin_enabled,
    sliders: [
      new SliderOptions({
        weight: 1.0,
        weightLabel: "Pin Weight",
        onWeightChange: set_pin_weight,
      }),
    ],
  }));

  // Stretch — stiffness OR compliance group, mutually exclusive
  const stretchStiffness = ConstraintGroup({
    label: 'stretch',
    enabled: true,
    onToggle: set_stretch_enabled,
    sliders: [
      new SliderOptions({
        weight: 0.5,
        weightLabel: "Stretching Weight",
        onWeightChange: set_stretch_weight,
      }),
    ],
  })
  const stretchCompliance = ConstraintGroup({
    label: 'stretch',
    enabled: true,
    onToggle: set_stretch_enabled,
    sliders: [
      new SliderOptions({
        weight: 7,
        weightLabel: 'Compliance (1e-x)',
        weightMin: 2, weightMax: 10, weightStep: 0.1,
        onWeightChange: v => set_stretch_compliance(Math.pow(10, -v)),
      }),
    ],
  })
  panel.appendChild(stretchStiffness)
  panel.appendChild(stretchCompliance)

  // Bending — stiffness OR compliance group
  const bendStiffness = ConstraintGroup({
    label: 'bending',
    enabled: true,
    onToggle: set_bending_enabled,
    sliders: [
      new SliderOptions({
        weight: 0.5,
        weightLabel: "Bending Weight",
        onWeightChange: set_bending_weight,
      }),
    ],
  })
  const bendCompliance = ConstraintGroup({
    label: 'bending',
    enabled: true,
    onToggle: set_bending_enabled,
    sliders: [
      new SliderOptions({
        weight: 6,
        weightLabel: 'Compliance (1e-x)',
        weightMin: 2, weightMax: 10, weightStep: 0.1,
        onWeightChange: v => set_bend_compliance(Math.pow(10, -v)),
      }),
    ],
  })
  panel.appendChild(bendStiffness)
  panel.appendChild(bendCompliance)

  panel.appendChild(ConstraintGroup({
    label: 'pulling',
    enabled: true,
    onToggle: set_pulling_enabled,
    sliders: [
      new SliderOptions({
        weight: 0.1,
        weightStep: 0.02,
        weightMax: 0.5,
        weightLabel: "Pulling Weight",
        onWeightChange: set_pulling_weight,
      }),
      new SliderOptions({
        weight: 5,
        weightStep: 1,
        weightMin: 0,
        weightMax: 20,
        weightLabel: "Pulling Area",
        onWeightChange: set_pulling_area,
      }),
    ],
  }));

  panel.appendChild(ConstraintGroup({
    label: 'self collision',
    enabled: true,
    onToggle: set_self_collision_enabled,
    sliders: [
      new SliderOptions({
        weight: 0.01,
        weightLabel: 'Collision Threshold',
        weightMin: 0.001,
        weightMax: 0.1,
        weightStep: 0.001,
        onWeightChange: set_self_collision_threshold,
      }),
      new SliderOptions({
        weight: 1,
        weightLabel: 'Recompute Iters',
        weightMin: 1,
        weightMax: 5,
        weightStep: 1,
        onWeightChange: set_self_collision_recompute_iters,
      })
    ],
  }));


  panel.appendChild(Divider());

  panel.appendChild(SliderRow({
    label: 'resolution', min: 4, max: 128, step: 1, value: 32,
    onChange: v => { currentResolution = v; set_resolution(Math.round(v)) },
  }));

  document.body.appendChild(panel);

  function applyDistanceConstraintsMode(on) {
    stretchStiffness.style.display = on ? 'none' : ''
    bendStiffness.style.display    = on ? 'none' : ''
    stretchCompliance.style.display = on ? '' : 'none'
    bendCompliance.style.display    = on ? '' : 'none'
    substepRow.style.display        = on ? '' : 'none'
  }
  applyDistanceConstraintsMode(USE_DC_DEFAULT)
}

init().then(async () => {
  set_use_distance_constraints(true);

  try {
    await run('canvas')
  } catch (e) {
    console.error('WASM run() failed:', e)
  }
  buildPanel()
})
