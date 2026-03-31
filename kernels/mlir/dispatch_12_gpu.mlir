// 12: [1024,2048] Q8_0 fused GEMV — pure MLIR GPU dialect
hal.executable public @decode_dispatch_12 {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "sm_87", features = "+ptx74", wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 166912, max_workgroup_counts = [2147483647, 65535, 65535]>>}>) {
    hal.executable.export public @decode_dispatch_12_matmul_1024x1x2048_f16 ordinal(0)
        layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>)
        count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      hal.return %c128, %c1, %c1 : index, index, index
    } attributes {workgroup_size = [256 : index, 1 : index, 1 : index]}
    builtin.module {
      func.func @decode_dispatch_12_matmul_1024x1x2048_f16()
          attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUBaseLowering workgroup_size = [256, 1, 1] subgroup_size = 32>} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c12352 = arith.constant 12352 : index
        %c2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c34 = arith.constant 34 : index
        %c256 = arith.constant 256 : index
        %c512 = arith.constant 512 : index
        %c768 = arith.constant 768 : index
        %c1024 = arith.constant 1024 : index
        %c1280 = arith.constant 1280 : index
        %c1536 = arith.constant 1536 : index
        %c1792 = arith.constant 1792 : index
        %c2048 = arith.constant 2048 : index
        %c2176 = arith.constant 2176 : index
        %c2228224 = arith.constant 2228224 : index
        %c8_i16 = arith.constant 8 : i16
        %cst_zero = arith.constant 0.000000e+00 : f32
        %c1_i32 = arith.constant 1 : i32
        %c2_i32 = arith.constant 2 : i32
        %c4_i32 = arith.constant 4 : i32
        %c8_i32 = arith.constant 8 : i32
        %c16_i32 = arith.constant 16 : i32
        %c32_i32 = arith.constant 32 : i32
        %c32_i64 = arith.constant 32 : i64

        %pc0 = hal.interface.constant.load layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
        %pc1 = hal.interface.constant.load layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
        %pc0_64 = arith.extui %pc0 : i32 to i64
        %pc1_64 = arith.extui %pc1 : i32 to i64
        %pc1_sh = arith.shli %pc1_64, %c32_i64 : i64
        %layer_i64 = arith.ori %pc0_64, %pc1_sh : i64
        %layer_idx = arith.index_castui %layer_i64 : i64 to index

        %c12352_bytes = arith.constant 12352 : index
        %c2048_bytes = arith.constant 2048 : index
        %q8 = hal.interface.binding.subspan layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<200000000xi8, #gpu.address_space<global>>
        %input = hal.interface.binding.subspan layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c12352_bytes) flags("ReadOnly|Indirect") : memref<2048xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        %output = hal.interface.binding.subspan layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c2048_bytes) flags(Indirect) : memref<1024xf16, strided<[1], offset: ?>, #gpu.address_space<global>>

        %tid = gpu.thread_id x upper_bound 256
        %bid = gpu.block_id x upper_bound 128

        %smem = memref.alloc() : memref<2048xf16, #gpu.address_space<workgroup>>
        %_sv0 = memref.load %input[%tid] : memref<2048xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %_sv0, %smem[%tid] : memref<2048xf16, #gpu.address_space<workgroup>>
        %_si1 = arith.addi %tid, %c256 : index
        %_sv1 = memref.load %input[%_si1] : memref<2048xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %_sv1, %smem[%_si1] : memref<2048xf16, #gpu.address_space<workgroup>>
        %_si2 = arith.addi %tid, %c512 : index
        %_sv2 = memref.load %input[%_si2] : memref<2048xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %_sv2, %smem[%_si2] : memref<2048xf16, #gpu.address_space<workgroup>>
        %_si3 = arith.addi %tid, %c768 : index
        %_sv3 = memref.load %input[%_si3] : memref<2048xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %_sv3, %smem[%_si3] : memref<2048xf16, #gpu.address_space<workgroup>>
        %_si4 = arith.addi %tid, %c1024 : index
        %_sv4 = memref.load %input[%_si4] : memref<2048xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %_sv4, %smem[%_si4] : memref<2048xf16, #gpu.address_space<workgroup>>
        %_si5 = arith.addi %tid, %c1280 : index
        %_sv5 = memref.load %input[%_si5] : memref<2048xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %_sv5, %smem[%_si5] : memref<2048xf16, #gpu.address_space<workgroup>>
        %_si6 = arith.addi %tid, %c1536 : index
        %_sv6 = memref.load %input[%_si6] : memref<2048xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %_sv6, %smem[%_si6] : memref<2048xf16, #gpu.address_space<workgroup>>
        %_si7 = arith.addi %tid, %c1792 : index
        %_sv7 = memref.load %input[%_si7] : memref<2048xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %_sv7, %smem[%_si7] : memref<2048xf16, #gpu.address_space<workgroup>>
        gpu.barrier

        %warp_id = arith.divui %tid, %c32 : index
        %lane = arith.remui %tid, %c32 : index
        %row_base = arith.muli %bid, %c8 : index
        %row = arith.addi %row_base, %warp_id : index

        %in_bounds = arith.cmpi ult, %row, %c1024 : index
        scf.if %in_bounds {
          %layer_off = arith.muli %layer_idx, %c2228224 : index
          %row_byte_off = arith.muli %row, %c2176 : index
          %row_byte_abs = arith.addi %layer_off, %row_byte_off : index



          %c64 = arith.constant 64 : index
          %scaled = scf.for %b = %lane to %c64 step %c32 iter_args(%acc = %cst_zero) -> (f32) {
            %b_off = arith.muli %b, %c34 : index
            %blk_byte = arith.addi %row_byte_abs, %b_off : index
            // Scale for this block
            %bs0 = memref.load %q8[%blk_byte] : memref<200000000xi8, #gpu.address_space<global>>
            %bs1_i = arith.addi %blk_byte, %c1 : index
            %bs1 = memref.load %q8[%bs1_i] : memref<200000000xi8, #gpu.address_space<global>>
            %bs0_16 = arith.extui %bs0 : i8 to i16
            %bs1_16 = arith.extui %bs1 : i8 to i16
            %bs1_sh = arith.shli %bs1_16, %c8_i16 : i16
            %bsc_16 = arith.ori %bs0_16, %bs1_sh : i16
            %bsc_f16 = arith.bitcast %bsc_16 : i16 to f16
            %bscale = arith.extf %bsc_f16 : f16 to f32
            %bv_base = arith.addi %blk_byte, %c2 : index
            %bk_base = arith.muli %b, %c32 : index
            %inner = scf.for %j = %c0 to %c32 step %c1 iter_args(%iacc = %cst_zero) -> (f32) {
              %j_off = arith.addi %bv_base, %j : index
              %qv = memref.load %q8[%j_off] : memref<200000000xi8, #gpu.address_space<global>>
              %qv_f = arith.sitofp %qv : i8 to f32
              %k = arith.addi %bk_base, %j : index
              %iv = memref.load %smem[%k] : memref<2048xf16, #gpu.address_space<workgroup>>
              %iv_f = arith.extf %iv : f16 to f32
              %p = arith.mulf %qv_f, %iv_f : f32
              %s = arith.addf %iacc, %p : f32
              scf.yield %s : f32
            }
            %blk_scaled = arith.mulf %bscale, %inner : f32
            %new_acc = arith.addf %acc, %blk_scaled : f32
            scf.yield %new_acc : f32
          }
          

          // Warp shuffle reduction
          %r1v, %r1ok = gpu.shuffle xor %scaled, %c16_i32, %c32_i32 : f32
          %r1 = arith.addf %scaled, %r1v : f32
          %r2v, %r2ok = gpu.shuffle xor %r1, %c8_i32, %c32_i32 : f32
          %r2 = arith.addf %r1, %r2v : f32
          %r3v, %r3ok = gpu.shuffle xor %r2, %c4_i32, %c32_i32 : f32
          %r3 = arith.addf %r2, %r3v : f32
          %r4v, %r4ok = gpu.shuffle xor %r3, %c2_i32, %c32_i32 : f32
          %r4 = arith.addf %r3, %r4v : f32
          %r5v, %r5ok = gpu.shuffle xor %r4, %c1_i32, %c32_i32 : f32
          %r5 = arith.addf %r4, %r5v : f32

          %is_lane0 = arith.cmpi eq, %lane, %c0 : index
          scf.if %is_lane0 {
            %out_f16 = arith.truncf %r5 : f32 to f16
            memref.store %out_f16, %output[%row] : memref<1024xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
          }
        }
        return
      }
    }
  }
}
