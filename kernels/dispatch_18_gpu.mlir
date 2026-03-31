// gate_up [6144,1024] Q8_0 fused GEMV — pure MLIR with GPU dialect
// Uses flat memref with manual offset arithmetic (matching CUDA kernel pattern)
hal.executable public @decode_dispatch_18 {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "sm_87", features = "+ptx74", wgp = <compute = fp64|fp32|fp16|int64|int32|int16|int8, storage = b64|b32|b16|b8, subgroup = shuffle|arithmetic, dot = dp4xi8toi32, mma = [], subgroup_size_choices = [32], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 166912, max_workgroup_counts = [2147483647, 65535, 65535]>>}>) {
    hal.executable.export public @decode_dispatch_18_matmul_6144x1x1024_f16 ordinal(0)
        layout(#hal.pipeline.layout<constants = 2, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
          #hal.pipeline.binding<storage_buffer, Indirect>
        ], flags = Indirect>)
        count(%arg0: !hal.device, %arg1: index, %arg2: index) -> (index, index, index) {
      %c768 = arith.constant 768 : index
      %c1 = arith.constant 1 : index
      hal.return %c768, %c1, %c1 : index, index, index
    } attributes {workgroup_size = [256 : index, 1 : index, 1 : index]}
    builtin.module {
      func.func @decode_dispatch_18_matmul_6144x1x1024_f16()
          attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUBaseLowering workgroup_size = [256, 1, 1] subgroup_size = 32>} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c768 = arith.constant 768 : index
        %c2 = arith.constant 2 : index
        %c8 = arith.constant 8 : index
        %c32 = arith.constant 32 : index
        %c34 = arith.constant 34 : index
        %c256 = arith.constant 256 : index
        %c512 = arith.constant 512 : index
        %c1024 = arith.constant 1024 : index
        %c1088 = arith.constant 1088 : index
        %c6144 = arith.constant 6144 : index
        %c6684672 = arith.constant 6684672 : index
        %c8_i16 = arith.constant 8 : i16
        %cst_zero = arith.constant 0.000000e+00 : f32
        %c1_i32 = arith.constant 1 : i32
        %c2_i32 = arith.constant 2 : i32
        %c4_i32 = arith.constant 4 : i32
        %c8_i32 = arith.constant 8 : i32
        %c16_i32 = arith.constant 16 : i32
        %c32_i32 = arith.constant 32 : i32
        %c32_i64 = arith.constant 32 : i64

        // Decode push constants: layer index
        %pc0 = hal.interface.constant.load layout(<constants = 2, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
          #hal.pipeline.binding<storage_buffer, Indirect>
        ], flags = Indirect>) ordinal(0) : i32
        %pc1 = hal.interface.constant.load layout(<constants = 2, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
          #hal.pipeline.binding<storage_buffer, Indirect>
        ], flags = Indirect>) ordinal(1) : i32
        %pc0_64 = arith.extui %pc0 : i32 to i64
        %pc1_64 = arith.extui %pc1 : i32 to i64
        %pc1_sh = arith.shli %pc1_64, %c32_i64 : i64
        %layer_i64 = arith.ori %pc0_64, %pc1_sh : i64
        %layer_idx = arith.index_castui %layer_i64 : i64 to index

        // Flat buffer bindings with #gpu.address_space<global> to skip cvta.to.global
        %c64_bytes = arith.constant 64 : index
        %c6144_bytes = arith.constant 6144 : index
        %q8 = hal.interface.binding.subspan layout(<constants = 2, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
          #hal.pipeline.binding<storage_buffer, Indirect>
        ], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<200000000xi8, #gpu.address_space<global>>
        %input = hal.interface.binding.subspan layout(<constants = 2, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
          #hal.pipeline.binding<storage_buffer, Indirect>
        ], flags = Indirect>) binding(1) alignment(64) offset(%c64_bytes) flags("ReadOnly|Indirect") : memref<1024xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        %output = hal.interface.binding.subspan layout(<constants = 2, bindings = [
          #hal.pipeline.binding<storage_buffer, ReadOnly>,
          #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
          #hal.pipeline.binding<storage_buffer, Indirect>
        ], flags = Indirect>) binding(2) alignment(64) offset(%c6144_bytes) flags(Indirect) : memref<6144xf16, strided<[1], offset: ?>, #gpu.address_space<global>>

        %tid = gpu.thread_id x upper_bound 256
        %bid = gpu.block_id x upper_bound 768

        // Shared memory — unrolled load (4 loads per thread for K=1024)
        %smem = memref.alloc() : memref<1024xf16, #gpu.address_space<workgroup>>
        %v0 = memref.load %input[%tid] : memref<1024xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %v0, %smem[%tid] : memref<1024xf16, #gpu.address_space<workgroup>>
        %i1 = arith.addi %tid, %c256 : index
        %v1 = memref.load %input[%i1] : memref<1024xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %v1, %smem[%i1] : memref<1024xf16, #gpu.address_space<workgroup>>
        %i2 = arith.addi %tid, %c512 : index
        %v2 = memref.load %input[%i2] : memref<1024xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %v2, %smem[%i2] : memref<1024xf16, #gpu.address_space<workgroup>>
        %i3 = arith.addi %tid, %c768 : index
        %v3 = memref.load %input[%i3] : memref<1024xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
        memref.store %v3, %smem[%i3] : memref<1024xf16, #gpu.address_space<workgroup>>
        gpu.barrier

        // Row = bid * 8 + tid / 32
        %warp_id = arith.divui %tid, %c32 : index
        %lane = arith.remui %tid, %c32 : index
        %row_base = arith.muli %bid, %c8 : index
        %row = arith.addi %row_base, %warp_id : index

        %in_bounds = arith.cmpi ult, %row, %c6144 : index
        scf.if %in_bounds {
          // Q8_0 flat byte offset: layer_idx * BPL + row * BYTES_PER_ROW + lane * 34
          %layer_off = arith.muli %layer_idx, %c6684672 : index
          %row_off = arith.muli %row, %c1088 : index
          %blk_off = arith.muli %lane, %c34 : index
          %off1 = arith.addi %layer_off, %row_off : index
          %byte_off = arith.addi %off1, %blk_off : index

          // Load scale
          %s0 = memref.load %q8[%byte_off] : memref<200000000xi8, #gpu.address_space<global>>
          %s1_idx = arith.addi %byte_off, %c1 : index
          %s1 = memref.load %q8[%s1_idx] : memref<200000000xi8, #gpu.address_space<global>>
          %s0_16 = arith.extui %s0 : i8 to i16
          %s1_16 = arith.extui %s1 : i8 to i16
          %s1_sh = arith.shli %s1_16, %c8_i16 : i16
          %sc_16 = arith.ori %s0_16, %s1_sh : i16
          %sc_f16 = arith.bitcast %sc_16 : i16 to f16
          %scale = arith.extf %sc_f16 : f16 to f32

          // Dot product
          %val_base = arith.addi %byte_off, %c2 : index
          %k_base = arith.muli %lane, %c32 : index
          %dot = scf.for %j = %c0 to %c32 step %c1 iter_args(%acc = %cst_zero) -> (f32) {
            %j_off = arith.addi %val_base, %j : index
            %qv = memref.load %q8[%j_off] : memref<200000000xi8, #gpu.address_space<global>>
            %qv_f = arith.sitofp %qv : i8 to f32
            %k = arith.addi %k_base, %j : index
            %iv = memref.load %smem[%k] : memref<1024xf16, #gpu.address_space<workgroup>>
            %iv_f = arith.extf %iv : f16 to f32
            %p = arith.mulf %qv_f, %iv_f : f32
            %s = arith.addf %acc, %p : f32
            scf.yield %s : f32
          }
          %scaled = arith.mulf %scale, %dot : f32

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
            memref.store %out_f16, %output[%row] : memref<6144xf16, strided<[1], offset: ?>, #gpu.address_space<global>>
          }
        }
        return
      }
    }
  }
}
