import './style.css'
import init, {
  run,
  set_time_step,
  set_constraint_iters,
  set_gravity_enabled,
  set_gravity_g,
  set_pin_enabled,
  set_pin_weight,
  set_stretch_enabled,
  set_stretch_weight,
  set_bending_enabled,
  set_bending_weight,
  set_pulling_enabled,
  set_pulling_weight,
  set_self_collision_enabled,
  set_self_collision_threshold,
  set_self_collision_recompute_pairs,
  set_use_distance_constraints,
  set_resolution,
  set_pulling_area,
} from '../rust/pkg/my_webgpu_app.js'
import {SliderRow, ConstraintGroup, CheckboxRow, Divider, SliderOptions} from './components.js'

function buildPanel() {
  const panel = document.createElement('div');
  panel.className = 'fixed top-4 left-4 z-10 bg-black/75 backdrop-blur-sm text-white rounded-xl p-4 w-60 text-xs font-mono select-none overflow-y-auto max-h-[calc(100vh-2rem)]';

  const title = document.createElement('div');
  title.className = 'text-sm font-bold mb-3 text-white/90 tracking-wide';
  title.textContent = 'Sim Params';
  panel.appendChild(title);

  panel.appendChild(SliderRow({
    label: 'time step', min: 0.001, max: 0.1, step: 0.001, value: 0.01,
    onChange: set_time_step,
  }));

  panel.appendChild(SliderRow({
    label: 'iterations', min: 1, max: 30, step: 1, value: 5,
    onChange: v => set_constraint_iters(Math.round(v)),
  }));

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
    checked: false,
    indent: false,
    onChange: set_use_distance_constraints,
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

  panel.appendChild(ConstraintGroup({
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
  }));

  panel.appendChild(ConstraintGroup({
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
  }));

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
    ],
  }));

  panel.appendChild(CheckboxRow({
    label: 'recompute pairs/iter',
    checked: false,
    onChange: set_self_collision_recompute_pairs,
  }));

  panel.appendChild(Divider());

  panel.appendChild(SliderRow({
    label: 'resolution', min: 4, max: 128, step: 1, value: 64,
    onChange: v => set_resolution(Math.round(v)),
  }));

  document.body.appendChild(panel);
}

init().then(async () => {
  try {
    await run('canvas')
  } catch (e) {
    console.error('WASM run() failed:', e)
  }
  buildPanel()
})
