import './style.css'
import init, { run } from '../rust/pkg/my_webgpu_app.js'

init().then(() => run('canvas'))
