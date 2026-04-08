pub struct PipelineBuilder<'a> {
    device: &'a wgpu::Device,
    shader_src: &'a str,
    vertex_layouts: Vec<wgpu::VertexBufferLayout<'a>>,
    bind_group_layouts: Vec<Option<&'a wgpu::BindGroupLayout>>,
    format: wgpu::TextureFormat,
    label: Option<&'a str>,
}

impl<'a> PipelineBuilder<'a> {
    pub fn new(device: &'a wgpu::Device, format: wgpu::TextureFormat) -> Self {
        Self {
            device,
            shader_src: "",
            vertex_layouts: vec![],
            bind_group_layouts: vec![],
            format,
            label: None,
        }
    }

    pub fn shader(mut self, src: &'a str) -> Self {
        self.shader_src = src;
        self
    }

    pub fn vertex_layout(mut self, layout: wgpu::VertexBufferLayout<'a>) -> Self {
        self.vertex_layouts.push(layout);
        self
    }

    pub fn bind_group_layout(mut self, layout: &'a wgpu::BindGroupLayout) -> Self {
        self.bind_group_layouts.push(Some(layout));
        self
    }

    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    pub fn build(self) -> wgpu::RenderPipeline {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: self.label,
            source: wgpu::ShaderSource::Wgsl(self.shader_src.into()),
        });

        let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: self.label,
            bind_group_layouts: &self.bind_group_layouts,
            immediate_size: 0,
        });

        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: self.label,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &self.vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        })
    }
}
