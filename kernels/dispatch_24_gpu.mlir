// dispatch_24: [151936,1024] f16 GEMV — pure MLIR GPU dialect
hal.executable public @decode_dispatch_24 {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "sm_87", features = "+ptx74", wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 166912, max_workgroup_counts = [2147483647, 65535, 65535]>>}>) {
    hal.executable.export public @decode_dispatch_24_matmul_151936x1x1024_f16 ordinal(0)
        layout(#hal.pipeline.layout<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>)
        count(%arg0: !hal.device) -> (index, index, index) {
      %c18992 = arith.constant 18992 : index
      %c1 = arith.constant 1 : index
      hal.return %c18992, %c1, %c1 : index, index, index
    } attributes {workgroup_size = [256 : index, 1 : index, 1 : index]}
    builtin.module {
      func.func @decode_dispatch_24_matmul_151936x1x1024_f16()
          attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUBaseLowering workgroup_size = [256, 1, 1] subgroup_size = 32>} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c256 = arith.constant 256 : index
        %c512 = arith.constant 512 : index
        %c768 = arith.constant 768 : index
        %c1024 = arith.constant 1024 : index
        %c151936 = arith.constant 151936 : index
        %cst_zero = arith.constant 0.000000e+00 : f32
        %c1_i32 = arith.constant 1 : i32
        %c2_i32 = arith.constant 2 : i32
        %c4_i32 = arith.constant 4 : i32
        %c8_i32 = arith.constant 8 : i32
        %c16_i32 = arith.constant 16 : i32
        %c32_i32 = arith.constant 32 : i32

        %pc0 = hal.interface.constant.load layout(#hal.pipeline.layout<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
        %out_off = arith.index_castui %pc0 : i32 to index

        %c64_bytes = arith.constant 64 : index
        %weight = hal.interface.binding.subspan layout(#hal.pipeline.layout<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<151936x1024xf16, #gpu.address_space<global>>
        %input = hal.interface.binding.subspan layout(#hal.pipeline.layout<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c64_bytes) flags("ReadOnly|Indirect") : memref<1024xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        %output = hal.interface.binding.subspan layout(#hal.pipeline.layout<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%out_off) flags(Indirect) : memref<151936xf16, strided<[1], offset: ?>, #gpu.address_space<global>>

        %tid = gpu.thread_id x upper_bound 256
        %bid = gpu.block_id x upper_bound 18992

        %smem = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
        %_sv0 = memref.load %input[%tid] : memref<1024xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %_sv0, %smem[%tid] : memref<1024xf16, #gpu.address_space<workgroup>>
        %_si1 = arith.addi %tid, %c256 : index
        %_sv1 = memref.load %input[%_si1] : memref<1024xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %_sv1, %smem[%_si1] : memref<1024xf16, #gpu.address_space<workgroup>>
        %_si2 = arith.addi %tid, %c512 : index
        %_sv2 = memref.load %input[%_si2] : memref<1024xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %_sv2, %smem[%_si2] : memref<1024xf16, #gpu.address_space<workgroup>>
        %_si3 = arith.addi %tid, %c768 : index
        %_sv3 = memref.load %input[%_si3] : memref<1024xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %_sv3, %smem[%_si3] : memref<1024xf16, #gpu.address_space<workgroup>>
        gpu.barrier

        %warp_id = arith.divui %tid, %c32 : index
        %lane = arith.remui %tid, %c32 : index
        %row_base = arith.muli %bid, %c8 : index
        %row = arith.addi %row_base, %warp_id : index

        %in_bounds = arith.cmpi ult, %row, %c151936 : index
        scf.if %in_bounds {
          // f16 dot product: each lane handles K/32 elements
          %dot = scf.for %k = %lane to %c1024 step %c32 iter_args(%acc = %cst_zero) -> (f32) {
            %wv = memref.load %weight[%row, %k] : memref<151936x1024xf16, #gpu.address_space<global>>
            %wv_f = arith.extf %wv : f16 to f32
            %iv = memref.load %smem[%k] : memref<1024xf16, #gpu.address_space<workgroup>>
            %iv_f = arith.extf %iv : f16 to f32
            %p = arith.mulf %wv_f, %iv_f : f32
            %s = arith.addf %acc, %p : f32
            scf.yield %s : f32
          }

          %r1v, %r1ok = gpu.shuffle xor %dot, %c16_i32, %c32_i32 : f32
          %r1 = arith.addf %dot, %r1v : f32
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
            memref.store %out_f16, %output[%row] : memref<151936xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
          }
        }
        return
      }
    }
  }
}
