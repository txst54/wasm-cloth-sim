/**
 * SimConsole — a fixed, non-interactive overlay that tails the paper sim's
 * verbose trace channel and auto-scrolls forever, like an `npm install` or
 * `cargo build` log. This is deliberately separate from console.log, which
 * stays throttled and spam-free.
 *
 * Position and size are driven entirely by CSS custom properties on
 * #sim-console (--sim-console-top / -left / -width / -height); override them in
 * style.css to move it wherever you want. The element is pointer-events: none,
 * so it never intercepts mouse input on the canvas.
 *
 * Usage:  startSimConsole(drain_paper_trace)   // pass the wasm export
 */

// How many trailing lines to keep in the DOM. The Rust side keeps its own
// (larger) ring buffer; this is just what stays on screen.
const MAX_LINES = 800

export function startSimConsole(drain) {
  if (document.getElementById('sim-console')) return

  const root = document.createElement('div')
  root.id = 'sim-console'
  const pre = document.createElement('pre')
  root.appendChild(pre)
  document.body.appendChild(root)

  const lines = []
  let dirty = false

  function pump() {
    const chunk = drain()
    if (chunk) {
      const incoming = chunk.split('\n')
      for (let i = 0; i < incoming.length; i++) lines.push(incoming[i])
      if (lines.length > MAX_LINES) lines.splice(0, lines.length - MAX_LINES)
      dirty = true
    }

    if (dirty) {
      pre.textContent = lines.join('\n')
      root.scrollTop = root.scrollHeight   // always pinned to the bottom
      dirty = false
    }

    requestAnimationFrame(pump)
  }

  requestAnimationFrame(pump)
}
