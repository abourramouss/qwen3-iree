hal.executable public @decode_dispatch_24 {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) objects([
    #hal.executable.object<{path = "dispatch_24.ptx"}>
  ]) {
    hal.executable.export public @decode_dispatch_24_matmul_151936x1x1024_f16 ordinal(0) layout(#hal.pipeline.layout<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %c18992 = arith.constant 18992 : index
      %c1 = arith.constant 1 : index
      hal.return %c18992, %c1, %c1 : index, index, index
    } attributes {workgroup_size = [256 : index, 1 : index, 1 : index]}
  }
}
