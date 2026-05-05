import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        paper: resolve(__dirname, 'paper/index.html'),
        particle: resolve(__dirname, 'particle/index.html'),
        particle_paper: resolve(__dirname, 'particle_paper/index.html'),
      },
    },
  },
})
