hal.executable public @decode_dispatch_12 {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb">) objects([
    #hal.executable.object<{path = "dispatch_12.ptx"}>
  ]) {
    hal.executable.export public @decode_dispatch_12_matmul_1024x1x2048_f16 ordinal(0) layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      hal.return %c128, %c1, %c1 : index, index, index
    } attributes {workgroup_size = [256 : index, 1 : index, 1 : index]}
  }
}
