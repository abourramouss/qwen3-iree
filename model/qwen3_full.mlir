#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> ()>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map16 = affine_map<(d0) -> ()>
module @llm_inference_qwen {
  // ---- Persistent state for multi-turn chat ----
  util.global private mutable @kv_cache = #util.uninitialized : !util.list<?>
  util.global private mutable @max_seq_len = 0 : index
  util.global private mutable @current_pos = 0 : index
  util.global private mutable @is_initialized = 0 : i1
  // ---- Fixed-size KV cache globals (no import/export sync needed) ----
  // 512 max tokens × 28 layers = 14336 slots × [8 heads, 128 dim] × f16
  util.global private mutable @g_k_cache = #util.uninitialized : tensor<14336x8x128xf16>
  util.global private mutable @g_v_cache = #util.uninitialized : tensor<14336x8x128xf16>

  util.func private @hparams.vocab_size() -> i64 {
    %c151936_i64 = arith.constant 151936 : i64
    util.return %c151936_i64 : i64
  }
  util.func private @hparams.block_count() -> i64 {
    %c28_i64 = arith.constant 28 : i64
    util.return %c28_i64 : i64
  }
  util.func private @hparams.embedding_length() -> i64 {
    %c1024_i64 = arith.constant 1024 : i64
    util.return %c1024_i64 : i64
  }
  util.func private @hparams.attention_head_count() -> i64 {
    %c16_i64 = arith.constant 16 : i64
    util.return %c16_i64 : i64
  }
  util.func private @hparams.attention_head_count_kv() -> i64 {
    %c8_i64 = arith.constant 8 : i64
    util.return %c8_i64 : i64
  }
  util.func private @hparams.feed_forward_length() -> i64 {
    %c3072_i64 = arith.constant 3072 : i64
    util.return %c3072_i64 : i64
  }
  util.func private @hparams.head_dim() -> i64 {
    %c128_i64 = arith.constant 128 : i64
    util.return %c128_i64 : i64
  }
  util.func private @hparams.rope_freq_base() -> f32 {
    %cst = arith.constant 1.000000e+06 : f32
    util.return %cst : f32
  }
  util.func private @hparams.layer_norm_rms_epsilon() -> f32 {
    %cst = arith.constant 9.99999997E-7 : f32
    util.return %cst : f32
  }
  util.func private @model_params.token_embd_weight() -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"token_embd.weight"> : tensor<151936x1024xf16>
    %cast = tensor.cast %0 : tensor<151936x1024xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.output_norm_weight() -> tensor<?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"output_norm.weight"> : tensor<1024xf16>
    %cast = tensor.cast %0 : tensor<1024xf16> to tensor<?xf16>
    util.return %cast : tensor<?xf16>
  }
  // output_weight: stored as [151936, 1024] (natural GGUF layout) for transposed GEMV
  util.func private @model_params.output_weight_T() -> tensor<151936x1024xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"output_T.weight"> : tensor<151936x1024xf16>
    util.return %0 : tensor<151936x1024xf16>
  }
  // Keep original for prefill path
  util.func private @model_params.output_weight() -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"output.weight"> : tensor<1024x151936xf16>
    %cast = tensor.cast %0 : tensor<1024x151936xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @embedding_components.embedding_lookup_1d(%arg0: tensor<?x?xf16>, %arg1: tensor<?xi64>) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?xi64>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %1 = iree_linalg_ext.gather dimension_map = [0] ins(%arg0, %arg1 : tensor<?x?xf16>, tensor<?xi64>) outs(%0 : tensor<?x?xf16>) -> tensor<?x?xf16>
    util.return %1 : tensor<?x?xf16>
  }
  // ---- Dynamic RMS norm (shared by prefill) ----
  util.func private @rms_norm_components.rms_norm_linalg(%arg0: tensor<?x?xf16>, %arg1: tensor<?xf16>, %arg2: f32) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %0 = tensor.empty(%dim) : tensor<?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf16>) outs(%1 : tensor<?xf32>) {
    ^bb0(%in: f16, %out: f32):
      %9 = arith.extf %in : f16 to f32
      %10 = arith.mulf %9, %9 : f32
      %11 = arith.addf %out, %10 : f32
      linalg.yield %11 : f32
    } -> tensor<?xf32>
    %3 = arith.index_cast %dim_0 : index to i32
    %4 = arith.sitofp %3 : i32 to f32
    %5 = tensor.empty(%dim) : tensor<?xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%2 : tensor<?xf32>) outs(%5 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.divf %in, %4 : f32
      %10 = arith.addf %9, %arg2 : f32
      %11 = math.sqrt %10 : f32
      linalg.yield %11 : f32
    } -> tensor<?xf32>
    %7 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %6, %arg1 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>) outs(%7 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f32, %in_2: f16, %out: f16):
      %9 = arith.extf %in : f16 to f32
      %10 = arith.extf %in_2 : f16 to f32
      %11 = arith.divf %9, %in_1 : f32
      %12 = arith.mulf %11, %10 : f32
      %13 = arith.truncf %12 : f32 to f16
      linalg.yield %13 : f16
    } -> tensor<?x?xf16>
    util.return %8 : tensor<?x?xf16>
  }
  // ---- Static RMS norm for decode: [1, 1024] ----
  util.func private @rms_norm_1x1024(%arg0: tensor<1x1024xf16>, %arg1: tensor<1024xf16>, %arg2: f32) -> tensor<1x1024xf16> {
    %0 = tensor.empty() : tensor<1xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1xf32>) -> tensor<1xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x1024xf16>) outs(%1 : tensor<1xf32>) {
    ^bb0(%in: f16, %out: f32):
      %9 = arith.extf %in : f16 to f32
      %10 = arith.mulf %9, %9 : f32
      %11 = arith.addf %out, %10 : f32
      linalg.yield %11 : f32
    } -> tensor<1xf32>
    %4 = arith.constant 1024.0 : f32
    %5 = tensor.empty() : tensor<1xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%2 : tensor<1xf32>) outs(%5 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.divf %in, %4 : f32
      %10 = arith.addf %9, %arg2 : f32
      %11 = math.sqrt %10 : f32
      linalg.yield %11 : f32
    } -> tensor<1xf32>
    %7 = tensor.empty() : tensor<1x1024xf16>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %6, %arg1 : tensor<1x1024xf16>, tensor<1xf32>, tensor<1024xf16>) outs(%7 : tensor<1x1024xf16>) {
    ^bb0(%in: f16, %in_1: f32, %in_2: f16, %out: f16):
      %9 = arith.extf %in : f16 to f32
      %10 = arith.extf %in_2 : f16 to f32
      %11 = arith.divf %9, %in_1 : f32
      %12 = arith.mulf %11, %10 : f32
      %13 = arith.truncf %12 : f32 to f16
      linalg.yield %13 : f16
    } -> tensor<1x1024xf16>
    util.return %8 : tensor<1x1024xf16>
  }
  util.func private @layer_norm_components.layer_norm(%arg0: tensor<?x?xf16>, %arg1: f32) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf16>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %dim_0 : index to i32
    %1 = arith.sitofp %0 : i32 to f32
    %2 = tensor.empty(%dim) : tensor<?xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf16>) outs(%3 : tensor<?xf32>) {
    ^bb0(%in: f16, %out: f32):
      %12 = arith.extf %in : f16 to f32
      %13 = arith.addf %out, %12 : f32
      linalg.yield %13 : f32
    } -> tensor<?xf32>
    %5 = tensor.empty(%dim) : tensor<?xf32>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%4 : tensor<?xf32>) outs(%5 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.divf %in, %1 : f32
      linalg.yield %12 : f32
    } -> tensor<?xf32>
    %7 = tensor.empty(%dim) : tensor<?xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<?xf32>) -> tensor<?xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0, %6 : tensor<?x?xf16>, tensor<?xf32>) outs(%8 : tensor<?xf32>) {
    ^bb0(%in: f16, %in_1: f32, %out: f32):
      %12 = arith.extf %in : f16 to f32
      %13 = arith.subf %12, %in_1 : f32
      %14 = arith.mulf %13, %13 : f32
      %15 = arith.addf %out, %14 : f32
      linalg.yield %15 : f32
    } -> tensor<?xf32>
    %10 = tensor.empty(%dim, %dim_0) : tensor<?x?xf16>
    %11 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %6, %9 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>) outs(%10 : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f32, %in_2: f32, %out: f16):
      %12 = arith.extf %in : f16 to f32
      %13 = arith.subf %12, %in_1 : f32
      %14 = arith.divf %in_2, %1 : f32
      %15 = arith.addf %14, %arg1 : f32
      %16 = math.sqrt %15 : f32
      %17 = arith.divf %13, %16 : f32
      %18 = arith.truncf %17 : f32 to f16
      linalg.yield %18 : f16
    } -> tensor<?x?xf16>
    util.return %11 : tensor<?x?xf16>
  }
  // ---- Direct (flat) KV cache: 3D [total_slots, n_head_kv, head_dim] ----
  util.func private @kvcache_components.allocate(%arg0: index, %arg1: index, %arg2: index) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f16
    %5 = tensor.empty(%arg0, %arg1, %arg2) : tensor<?x?x?xf16>
    %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
    %7 = hal.tensor.export %6 : tensor<?x?x?xf16>{%arg0, %arg1, %arg2} -> !hal.buffer_view
    %cst_v = arith.constant 0.000000e+00 : f16
    %8 = tensor.empty(%arg0, %arg1, %arg2) : tensor<?x?x?xf16>
    %9 = linalg.fill ins(%cst_v : f16) outs(%8 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
    %10 = hal.tensor.export %9 : tensor<?x?x?xf16>{%arg0, %arg1, %arg2} -> !hal.buffer_view
    %11 = util.list.create %c2 : !util.list<?>
    util.list.resize %11, %c2 : !util.list<?>
    util.list.set %11[%c0], %7 : !hal.buffer_view -> !util.list<?>
    util.list.set %11[%c1], %10 : !hal.buffer_view -> !util.list<?>
    util.return %11 : !util.list<?>
  }
  // ---- Static decode KV cache read: output [1, ctx_len, 8, 128] ----
  util.func private @kvcache_direct_read_static(%cache: !util.list<?>, %slot_start: index, %ctx_len: index) -> (tensor<1x?x8x128xf16>, tensor<1x?x8x128xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view
    %total = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %k_full = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x8x128xf16>{%total}
    %v_full = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x8x128xf16>{%total}
    %ks = tensor.extract_slice %k_full[%slot_start, 0, 0] [%ctx_len, 8, 128] [1, 1, 1] : tensor<?x8x128xf16> to tensor<?x8x128xf16>
    %vs = tensor.extract_slice %v_full[%slot_start, 0, 0] [%ctx_len, 8, 128] [1, 1, 1] : tensor<?x8x128xf16> to tensor<?x8x128xf16>
    %ko = tensor.expand_shape %ks [[0, 1], [2], [3]] output_shape [1, %ctx_len, 8, 128] : tensor<?x8x128xf16> into tensor<1x?x8x128xf16>
    %vo = tensor.expand_shape %vs [[0, 1], [2], [3]] output_shape [1, %ctx_len, 8, 128] : tensor<?x8x128xf16> into tensor<1x?x8x128xf16>
    util.return %ko, %vo : tensor<1x?x8x128xf16>, tensor<1x?x8x128xf16>
  }
  // ---- Static decode KV cache write: insert [8, 128] slice ----
  util.func private @kvcache_direct_write_static(%cache: !util.list<?>, %slot: index, %new_k: tensor<8x128xf16>, %new_v: tensor<8x128xf16>) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view
    %total = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %k_full = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x8x128xf16>{%total}
    %v_full = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x8x128xf16>{%total}
    %ku = tensor.insert_slice %new_k into %k_full[%slot, 0, 0] [1, 8, 128] [1, 1, 1] : tensor<8x128xf16> into tensor<?x8x128xf16>
    %vu = tensor.insert_slice %new_v into %v_full[%slot, 0, 0] [1, 8, 128] [1, 1, 1] : tensor<8x128xf16> into tensor<?x8x128xf16>
    %kbv = hal.tensor.export %ku : tensor<?x8x128xf16>{%total} -> !hal.buffer_view
    %vbv = hal.tensor.export %vu : tensor<?x8x128xf16>{%total} -> !hal.buffer_view
    util.list.set %cache[%c0], %kbv : !hal.buffer_view -> !util.list<?>
    util.list.set %cache[%c1], %vbv : !hal.buffer_view -> !util.list<?>
    util.return %cache : !util.list<?>
  }
  // ---- Dynamic KV cache read (for prefill path) ----
  util.func private @kvcache_direct_read(%cache: !util.list<?>, %slot_start: index, %ctx_len: index) -> (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view
    %total = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %nkv = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %hdim = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[2] : index
    %k_full = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total, %nkv, %hdim}
    %v_full = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total, %nkv, %hdim}
    %ks = tensor.extract_slice %k_full[%slot_start, 0, 0] [%ctx_len, %nkv, %hdim] [1, 1, 1] : tensor<?x?x?xf16> to tensor<?x?x?xf16>
    %vs = tensor.extract_slice %v_full[%slot_start, 0, 0] [%ctx_len, %nkv, %hdim] [1, 1, 1] : tensor<?x?x?xf16> to tensor<?x?x?xf16>
    %ko = tensor.expand_shape %ks [[0, 1], [2], [3]] output_shape [%c1, %ctx_len, %nkv, %hdim] : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %vo = tensor.expand_shape %vs [[0, 1], [2], [3]] output_shape [%c1, %ctx_len, %nkv, %hdim] : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    util.return %ko, %vo : tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>
  }
  // ---- Dynamic KV cache write (for prefill path) ----
  util.func private @kvcache_direct_write(%cache: !util.list<?>, %slot: index, %new_k: tensor<?x?xf16>, %new_v: tensor<?x?xf16>) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view
    %total = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %nkv = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %hdim = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[2] : index
    %k_full = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total, %nkv, %hdim}
    %v_full = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total, %nkv, %hdim}
    %ku = tensor.insert_slice %new_k into %k_full[%slot, 0, 0] [1, %nkv, %hdim] [1, 1, 1] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %vu = tensor.insert_slice %new_v into %v_full[%slot, 0, 0] [1, %nkv, %hdim] [1, 1, 1] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %kbv = hal.tensor.export %ku : tensor<?x?x?xf16>{%total, %nkv, %hdim} -> !hal.buffer_view
    %vbv = hal.tensor.export %vu : tensor<?x?x?xf16>{%total, %nkv, %hdim} -> !hal.buffer_view
    util.list.set %cache[%c0], %kbv : !hal.buffer_view -> !util.list<?>
    util.list.set %cache[%c1], %vbv : !hal.buffer_view -> !util.list<?>
    util.return %cache : !util.list<?>
  }
  // ---- Direct KV cache scatter for prefill: insert_slice for a batch of tokens ----
  util.func private @kvcache_direct_scatter_prefill(%cache: !util.list<?>, %layer_idx: index, %new_k: tensor<?x?x?xf16>, %new_v: tensor<?x?x?xf16>, %max_seq_len_val: index, %start_pos: index) -> !util.list<?> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %seq_len = tensor.dim %new_k, %c0 : tensor<?x?x?xf16>
    %slot_start_base = arith.muli %layer_idx, %max_seq_len_val : index
    %slot_start = arith.addi %slot_start_base, %start_pos : index
    %k_bv = util.list.get %cache[%c0] : !util.list<?> -> !hal.buffer_view
    %v_bv = util.list.get %cache[%c1] : !util.list<?> -> !hal.buffer_view
    %total = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[0] : index
    %nkv = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[1] : index
    %hdim = hal.buffer_view.dim<%k_bv : !hal.buffer_view>[2] : index
    %k_full = hal.tensor.import %k_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total, %nkv, %hdim}
    %v_full = hal.tensor.import %v_bv : !hal.buffer_view -> tensor<?x?x?xf16>{%total, %nkv, %hdim}
    %ku = tensor.insert_slice %new_k into %k_full[%slot_start, 0, 0] [%seq_len, %nkv, %hdim] [1, 1, 1] : tensor<?x?x?xf16> into tensor<?x?x?xf16>
    %vu = tensor.insert_slice %new_v into %v_full[%slot_start, 0, 0] [%seq_len, %nkv, %hdim] [1, 1, 1] : tensor<?x?x?xf16> into tensor<?x?x?xf16>
    %kbv = hal.tensor.export %ku : tensor<?x?x?xf16>{%total, %nkv, %hdim} -> !hal.buffer_view
    %vbv = hal.tensor.export %vu : tensor<?x?x?xf16>{%total, %nkv, %hdim} -> !hal.buffer_view
    util.list.set %cache[%c0], %kbv : !hal.buffer_view -> !util.list<?>
    util.list.set %cache[%c1], %vbv : !hal.buffer_view -> !util.list<?>
    util.return %cache : !util.list<?>
  }
  // ---- Static RoPE for decode Q: [1, 1, 16, 128] ----
  util.func private @rope_decode_q(%arg0: tensor<1x1x16x128xf16>, %arg1: tensor<1x1xi64>) -> tensor<1x1x16x128xf16> {
    %c64 = arith.constant 64 : index
    %cst_dim = arith.constant 128.0 : f32
    %cst_base = arith.constant 1.000000e+06 : f32
    %cst_scale = arith.constant 1.000000e+00 : f32
    %3 = tensor.empty() : tensor<1x1x16x128xf16>
    %4 = linalg.generic {indexing_maps = [#map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3 : tensor<1x1x16x128xf16>) {
    ^bb0(%out: f16):
      %5 = linalg.index 0 : index
      %6 = linalg.index 1 : index
      %7 = linalg.index 2 : index
      %8 = linalg.index 3 : index
      %extracted = tensor.extract %arg1[%5, %6] : tensor<1x1xi64>
      %9 = arith.trunci %extracted : i64 to i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.cmpi slt, %8, %c64 : index
      %12 = scf.if %11 -> (f16) {
        %13 = arith.index_cast %8 : index to i32
        %14 = arith.sitofp %13 : i32 to f32
        %cst = arith.constant 2.000000e+00 : f32
        %15 = arith.mulf %cst, %14 : f32
        %16 = arith.divf %15, %cst_dim : f32
        %17 = arith.negf %16 : f32
        %18 = math.powf %cst_base, %17 : f32
        %19 = arith.mulf %18, %cst_scale : f32
        %20 = arith.mulf %10, %19 : f32
        %21 = math.cos %20 : f32
        %22 = math.sin %20 : f32
        %extracted_3 = tensor.extract %arg0[%5, %6, %7, %8] : tensor<1x1x16x128xf16>
        %23 = arith.addi %8, %c64 : index
        %extracted_4 = tensor.extract %arg0[%5, %6, %7, %23] : tensor<1x1x16x128xf16>
        %24 = arith.extf %extracted_3 : f16 to f32
        %25 = arith.extf %extracted_4 : f16 to f32
        %26 = arith.mulf %24, %21 : f32
        %27 = arith.mulf %25, %22 : f32
        %28 = arith.subf %26, %27 : f32
        %29 = arith.truncf %28 : f32 to f16
        scf.yield %29 : f16
      } else {
        %13 = arith.subi %8, %c64 : index
        %14 = arith.index_cast %13 : index to i32
        %15 = arith.sitofp %14 : i32 to f32
        %cst = arith.constant 2.000000e+00 : f32
        %16 = arith.mulf %cst, %15 : f32
        %17 = arith.divf %16, %cst_dim : f32
        %18 = arith.negf %17 : f32
        %19 = math.powf %cst_base, %18 : f32
        %20 = arith.mulf %19, %cst_scale : f32
        %21 = arith.mulf %10, %20 : f32
        %22 = math.cos %21 : f32
        %23 = math.sin %21 : f32
        %extracted_3 = tensor.extract %arg0[%5, %6, %7, %8] : tensor<1x1x16x128xf16>
        %extracted_4 = tensor.extract %arg0[%5, %6, %7, %13] : tensor<1x1x16x128xf16>
        %24 = arith.extf %extracted_4 : f16 to f32
        %25 = arith.extf %extracted_3 : f16 to f32
        %26 = arith.mulf %24, %23 : f32
        %27 = arith.mulf %25, %22 : f32
        %28 = arith.addf %26, %27 : f32
        %29 = arith.truncf %28 : f32 to f16
        scf.yield %29 : f16
      }
      linalg.yield %12 : f16
    } -> tensor<1x1x16x128xf16>
    util.return %4 : tensor<1x1x16x128xf16>
  }
  // ---- Static RoPE for decode K: [1, 1, 8, 128] ----
  util.func private @rope_decode_k(%arg0: tensor<1x1x8x128xf16>, %arg1: tensor<1x1xi64>) -> tensor<1x1x8x128xf16> {
    %c64 = arith.constant 64 : index
    %cst_dim = arith.constant 128.0 : f32
    %cst_base = arith.constant 1.000000e+06 : f32
    %cst_scale = arith.constant 1.000000e+00 : f32
    %3 = tensor.empty() : tensor<1x1x8x128xf16>
    %4 = linalg.generic {indexing_maps = [#map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3 : tensor<1x1x8x128xf16>) {
    ^bb0(%out: f16):
      %5 = linalg.index 0 : index
      %6 = linalg.index 1 : index
      %7 = linalg.index 2 : index
      %8 = linalg.index 3 : index
      %extracted = tensor.extract %arg1[%5, %6] : tensor<1x1xi64>
      %9 = arith.trunci %extracted : i64 to i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.cmpi slt, %8, %c64 : index
      %12 = scf.if %11 -> (f16) {
        %13 = arith.index_cast %8 : index to i32
        %14 = arith.sitofp %13 : i32 to f32
        %cst = arith.constant 2.000000e+00 : f32
        %15 = arith.mulf %cst, %14 : f32
        %16 = arith.divf %15, %cst_dim : f32
        %17 = arith.negf %16 : f32
        %18 = math.powf %cst_base, %17 : f32
        %19 = arith.mulf %18, %cst_scale : f32
        %20 = arith.mulf %10, %19 : f32
        %21 = math.cos %20 : f32
        %22 = math.sin %20 : f32
        %extracted_3 = tensor.extract %arg0[%5, %6, %7, %8] : tensor<1x1x8x128xf16>
        %23 = arith.addi %8, %c64 : index
        %extracted_4 = tensor.extract %arg0[%5, %6, %7, %23] : tensor<1x1x8x128xf16>
        %24 = arith.extf %extracted_3 : f16 to f32
        %25 = arith.extf %extracted_4 : f16 to f32
        %26 = arith.mulf %24, %21 : f32
        %27 = arith.mulf %25, %22 : f32
        %28 = arith.subf %26, %27 : f32
        %29 = arith.truncf %28 : f32 to f16
        scf.yield %29 : f16
      } else {
        %13 = arith.subi %8, %c64 : index
        %14 = arith.index_cast %13 : index to i32
        %15 = arith.sitofp %14 : i32 to f32
        %cst = arith.constant 2.000000e+00 : f32
        %16 = arith.mulf %cst, %15 : f32
        %17 = arith.divf %16, %cst_dim : f32
        %18 = arith.negf %17 : f32
        %19 = math.powf %cst_base, %18 : f32
        %20 = arith.mulf %19, %cst_scale : f32
        %21 = arith.mulf %10, %20 : f32
        %22 = math.cos %21 : f32
        %23 = math.sin %21 : f32
        %extracted_3 = tensor.extract %arg0[%5, %6, %7, %8] : tensor<1x1x8x128xf16>
        %extracted_4 = tensor.extract %arg0[%5, %6, %7, %13] : tensor<1x1x8x128xf16>
        %24 = arith.extf %extracted_4 : f16 to f32
        %25 = arith.extf %extracted_3 : f16 to f32
        %26 = arith.mulf %24, %23 : f32
        %27 = arith.mulf %25, %22 : f32
        %28 = arith.addf %26, %27 : f32
        %29 = arith.truncf %28 : f32 to f16
        scf.yield %29 : f16
      }
      linalg.yield %12 : f16
    } -> tensor<1x1x8x128xf16>
    util.return %4 : tensor<1x1x8x128xf16>
  }
  // ---- Static GQA attention for decode: Q=[1,1,16,128], K/V=[1,?,8,128] ----
  util.func private @attention_gqa_decode_static(
      %arg0: tensor<1x1x16x128xf16>,
      %arg1: tensor<1x?x8x128xf16>,
      %arg2: tensor<1x?x8x128xf16>,
      %arg3: f32) -> tensor<1x1x16x128xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // seq_len_kv is dynamic (ctx_len+1)
    %dim_4 = tensor.dim %arg1, %c1 : tensor<1x?x8x128xf16>
    // Transpose Q: [1, seq_q=1, heads=16, dim=128] -> [1, heads=16, seq_q=1, dim=128]
    %0 = tensor.empty() : tensor<1x16x1x128xf16>
    %1 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1x16x128xf16>) outs(%0 : tensor<1x16x1x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x16x1x128xf16>
    // Transpose K: [1, seq_kv, 8, 128] -> [1, 8, seq_kv, 128]
    %2 = tensor.empty(%dim_4) : tensor<1x8x?x128xf16>
    %3 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x?x8x128xf16>) outs(%2 : tensor<1x8x?x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x8x?x128xf16>
    // Transpose V: [1, seq_kv, 8, 128] -> [1, 8, seq_kv, 128]
    %4 = tensor.empty(%dim_4) : tensor<1x8x?x128xf16>
    %5 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x?x8x128xf16>) outs(%4 : tensor<1x8x?x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x8x?x128xf16>
    // GQA expand K: [1, 8, seq_kv, 128] -> [1, 8, 2, seq_kv, 128] -> collapse -> [1, 16, seq_kv, 128]
    %7 = tensor.empty(%dim_4) : tensor<1x8x2x?x128xf16>
    %8 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x8x?x128xf16>) outs(%7 : tensor<1x8x2x?x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x8x2x?x128xf16>
    %collapsed = tensor.collapse_shape %8 [[0], [1, 2], [3], [4]] : tensor<1x8x2x?x128xf16> into tensor<1x16x?x128xf16>
    // GQA expand V: same
    %9 = tensor.empty(%dim_4) : tensor<1x8x2x?x128xf16>
    %10 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%5 : tensor<1x8x?x128xf16>) outs(%9 : tensor<1x8x2x?x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x8x2x?x128xf16>
    %collapsed_5 = tensor.collapse_shape %10 [[0], [1, 2], [3], [4]] : tensor<1x8x2x?x128xf16> into tensor<1x16x?x128xf16>
    // Collapse batch*heads: [1, 16, seq, 128] -> [16, seq, 128]
    %collapsed_6 = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x16x1x128xf16> into tensor<16x1x128xf16>
    %collapsed_7 = tensor.collapse_shape %collapsed [[0, 1], [2], [3]] : tensor<1x16x?x128xf16> into tensor<16x?x128xf16>
    %collapsed_8 = tensor.collapse_shape %collapsed_5 [[0, 1], [2], [3]] : tensor<1x16x?x128xf16> into tensor<16x?x128xf16>
    // Causal mask: [16, 1, seq_kv] — for decode seq_q=1, so mask[b, 0, j] = (j <= 0 + offset)
    // offset = seq_kv - seq_q = dim_4 - 1
    %c1_idx = arith.constant 1 : index
    %12 = arith.subi %dim_4, %c1_idx : index
    %c16 = arith.constant 16 : index
    %13 = tensor.empty(%dim_4) : tensor<16x1x?xi1>
    %14 = linalg.generic {indexing_maps = [#map8], iterator_types = ["parallel", "parallel", "parallel"]} outs(%13 : tensor<16x1x?xi1>) {
    ^bb0(%out: i1):
      %19 = linalg.index 1 : index
      %20 = linalg.index 2 : index
      %21 = arith.addi %19, %12 : index
      %22 = arith.cmpi ule, %20, %21 : index
      linalg.yield %22 : i1
    } -> tensor<16x1x?xi1>
    // Attention
    %15 = tensor.empty() : tensor<16x1x128xf16>
    %16 = iree_linalg_ext.attention {indexing_maps = [#map9, #map10, #map11, #map12, #map13, #map14]} ins(%collapsed_6, %collapsed_7, %collapsed_8, %arg3, %14 : tensor<16x1x128xf16>, tensor<16x?x128xf16>, tensor<16x?x128xf16>, f32, tensor<16x1x?xi1>) outs(%15 : tensor<16x1x128xf16>) {
    ^bb0(%arg4: f32):
      iree_linalg_ext.yield %arg4 : f32
    } -> tensor<16x1x128xf16>
    // Reshape back: [16, 1, 128] -> [1, 16, 1, 128]
    %expanded = tensor.expand_shape %16 [[0, 1], [2], [3]] output_shape [1, 16, 1, 128] : tensor<16x1x128xf16> into tensor<1x16x1x128xf16>
    // Transpose back: [1, 16, 1, 128] -> [1, 1, 16, 128]
    %17 = tensor.empty() : tensor<1x1x16x128xf16>
    %18 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x16x1x128xf16>) outs(%17 : tensor<1x1x16x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1x1x16x128xf16>
    util.return %18 : tensor<1x1x16x128xf16>
  }
  // ---- Q8_0 fused dequant+GEMV helper ----
  // Fused Q8_0 dequant + transposed GEMV: Q8[N,K] @ input[K,1] -> output[N,1]
  // Uses flow.dispatch.region to force single dispatch (no separate dequant pass).
  // q8_stacked: tensor<28 x bytes_per_layer x i8>, input_t: tensor<K x 1 x f16>
  // Returns: tensor<N x 1 x f16>
  util.func private @q8_fused_gemv(
      %q8_stacked: tensor<28x?xi8>,
      %layer: i32,
      %input_t: tensor<?x1xf16>,
      %N: index, %K: index
  ) -> tensor<?x1xf16> {
    %c1 = arith.constant 1 : index
    %bpl = tensor.dim %q8_stacked, %c1 : tensor<28x?xi8>
    %result = flow.dispatch.region[] -> (tensor<?x1xf16>{%N}) {
      %_layer_idx = arith.index_cast %layer : i32 to index
      %_raw_slice = tensor.extract_slice %q8_stacked[%_layer_idx, 0] [1, %bpl] [1, 1]
          : tensor<28x?xi8> to tensor<1x?xi8>
      %_raw = tensor.collapse_shape %_raw_slice [[0, 1]] : tensor<1x?xi8> into tensor<?xi8>
      %_c1 = arith.constant 1 : index
      %_c2 = arith.constant 2 : index
      %_c8_i16 = arith.constant 8 : i16
      %_c32 = arith.constant 32 : index
      %_c34 = arith.constant 34 : index
      %_w_init = tensor.empty(%N, %K) : tensor<?x?xf16>
      %_w = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } outs(%_w_init : tensor<?x?xf16>) {
      ^bb0(%out: f16):
        %k = linalg.index 0 : index
        %n = linalg.index 1 : index
        %kN = arith.muli %k, %K : index
        %flat_idx = arith.addi %kN, %n : index
        %blk = arith.divui %flat_idx, %_c32 : index
        %elem = arith.remui %flat_idx, %_c32 : index
        %boff = arith.muli %blk, %_c34 : index
        %s1_off = arith.addi %boff, %_c1 : index
        %s0 = tensor.extract %_raw[%boff] : tensor<?xi8>
        %s1 = tensor.extract %_raw[%s1_off] : tensor<?xi8>
        %s0_i16 = arith.extui %s0 : i8 to i16
        %s1_i16 = arith.extui %s1 : i8 to i16
        %s1_sh = arith.shli %s1_i16, %_c8_i16 : i16
        %scale_i16 = arith.ori %s0_i16, %s1_sh : i16
        %scale = arith.bitcast %scale_i16 : i16 to f16
        %v_off = arith.addi %boff, %_c2 : index
        %e_off = arith.addi %v_off, %elem : index
        %qval = tensor.extract %_raw[%e_off] : tensor<?xi8>
        %qval_f16 = arith.sitofp %qval : i8 to f16
        %dq = arith.mulf %scale, %qval_f16 : f16
        linalg.yield %dq : f16
      } -> tensor<?x?xf16>
      %_cst = arith.constant 0.000000e+00 : f16
      %_out_init = tensor.empty(%N) : tensor<?x1xf16>
      %_out_zero = linalg.fill ins(%_cst : f16) outs(%_out_init : tensor<?x1xf16>) -> tensor<?x1xf16>
      %_result = linalg.matmul ins(%_w, %input_t : tensor<?x?xf16>, tensor<?x1xf16>) outs(%_out_zero : tensor<?x1xf16>) -> tensor<?x1xf16>
      flow.return %_result : tensor<?x1xf16>
    }
    util.return %result : tensor<?x1xf16>
  }
  // ---- Static attention block for decode (fused Q8_0 dequant + GEMV) ----
  util.func private @attention_block_decode_static(
      %arg0: tensor<1x1024xf16>,         // normed hidden [1, 1024]
      %arg1: tensor<1x1xi64>,             // positions [1, 1]
      %arg2: tensor<1x?x8x128xf16>,      // cached K [1, ctx_len, 8, 128]
      %arg3: tensor<1x?x8x128xf16>,      // cached V [1, ctx_len, 8, 128]
      %q8_q: tensor<28x2228224xi8>,       // Q8_0 stacked Q weight
      %q8_k: tensor<28x1114112xi8>,       // Q8_0 stacked K weight
      %q8_v: tensor<28x1114112xi8>,       // Q8_0 stacked V weight
      %q8_o: tensor<28x2228224xi8>,       // Q8_0 stacked O weight
      %w_qn: tensor<128xf16>,             // Q norm weight
      %w_kn: tensor<128xf16>,             // K norm weight
      %layer_i32: i32                     // layer index
  ) -> (tensor<1x1024xf16>, tensor<8x128xf16>, tensor<8x128xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %eps = arith.constant 9.99999997E-7 : f32
    %dim_0 = tensor.dim %arg2, %c1 : tensor<1x?x8x128xf16>
    // QKV fused Q8_0 dequant + transposed GEMV
    // Transpose input once: [1, 1024] -> [1024, 1]
    %x_t_init = tensor.empty() : tensor<1024x1xf16>
    %x_t = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<1x1024xf16>) outs(%x_t_init : tensor<1024x1xf16>) {
    ^bb0(%in: f16, %out: f16): linalg.yield %in : f16
    } -> tensor<1024x1xf16>
    %x_t_dyn = tensor.cast %x_t : tensor<1024x1xf16> to tensor<?x1xf16>
    %q8_q_dyn = tensor.cast %q8_q : tensor<28x2228224xi8> to tensor<28x?xi8>
    %q8_k_dyn = tensor.cast %q8_k : tensor<28x1114112xi8> to tensor<28x?xi8>
    %q8_v_dyn = tensor.cast %q8_v : tensor<28x1114112xi8> to tensor<28x?xi8>
    %c2048 = arith.constant 2048 : index
    %c1024 = arith.constant 1024 : index
    // Q: [2048, 1024] @ [1024, 1] -> [2048, 1] -> [1, 2048]
    %q_mm_dyn = util.call @q8_fused_gemv(%q8_q_dyn, %layer_i32, %x_t_dyn, %c2048, %c1024) : (tensor<28x?xi8>, i32, tensor<?x1xf16>, index, index) -> tensor<?x1xf16>
    %q_mm = tensor.cast %q_mm_dyn : tensor<?x1xf16> to tensor<2048x1xf16>
    %q_proj_flat = tensor.collapse_shape %q_mm [[0, 1]] : tensor<2048x1xf16> into tensor<2048xf16>
    %q_proj = tensor.expand_shape %q_proj_flat [[0, 1]] output_shape [1, 2048] : tensor<2048xf16> into tensor<1x2048xf16>
    // K: [1024, 1024] @ [1024, 1] -> [1024, 1] -> [1, 1024]
    %k_mm_dyn = util.call @q8_fused_gemv(%q8_k_dyn, %layer_i32, %x_t_dyn, %c1024, %c1024) : (tensor<28x?xi8>, i32, tensor<?x1xf16>, index, index) -> tensor<?x1xf16>
    %k_mm = tensor.cast %k_mm_dyn : tensor<?x1xf16> to tensor<1024x1xf16>
    %k_proj_flat = tensor.collapse_shape %k_mm [[0, 1]] : tensor<1024x1xf16> into tensor<1024xf16>
    %k_proj = tensor.expand_shape %k_proj_flat [[0, 1]] output_shape [1, 1024] : tensor<1024xf16> into tensor<1x1024xf16>
    // V: [1024, 1024] @ [1024, 1] -> [1024, 1] -> [1, 1024]
    %v_mm_dyn = util.call @q8_fused_gemv(%q8_v_dyn, %layer_i32, %x_t_dyn, %c1024, %c1024) : (tensor<28x?xi8>, i32, tensor<?x1xf16>, index, index) -> tensor<?x1xf16>
    %v_mm = tensor.cast %v_mm_dyn : tensor<?x1xf16> to tensor<1024x1xf16>
    %v_proj_flat = tensor.collapse_shape %v_mm [[0, 1]] : tensor<1024x1xf16> into tensor<1024xf16>
    %v_proj = tensor.expand_shape %v_proj_flat [[0, 1]] output_shape [1, 1024] : tensor<1024xf16> into tensor<1x1024xf16>
    // Reshape to [1, heads, head_dim] then [batch*heads, head_dim] for QK norm
    // Q: [1, 2048] -> [1, 16, 128] -> collapse [16, 128]
    %q_3d = tensor.expand_shape %q_proj [[0], [1, 2]] output_shape [1, 16, 128] : tensor<1x2048xf16> into tensor<1x16x128xf16>
    %k_3d = tensor.expand_shape %k_proj [[0], [1, 2]] output_shape [1, 8, 128] : tensor<1x1024xf16> into tensor<1x8x128xf16>
    %v_3d = tensor.expand_shape %v_proj [[0], [1, 2]] output_shape [1, 8, 128] : tensor<1x1024xf16> into tensor<1x8x128xf16>
    %q_flat = tensor.collapse_shape %q_3d [[0, 1], [2]] : tensor<1x16x128xf16> into tensor<16x128xf16>
    %k_flat = tensor.collapse_shape %k_3d [[0, 1], [2]] : tensor<1x8x128xf16> into tensor<8x128xf16>
    // QK norm via flow.dispatch.region
    // Q norm: [16, 128]
    %q_normed = flow.dispatch.region[] -> (tensor<16x128xf16>) {
      %_cst0 = arith.constant 0.000000e+00 : f32
      %_hdf = arith.constant 128.0 : f32
      %_si = tensor.empty() : tensor<16xf32>
      %_sz = linalg.fill ins(%_cst0 : f32) outs(%_si : tensor<16xf32>) -> tensor<16xf32>
      %_ss = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%q_flat : tensor<16x128xf16>) outs(%_sz : tensor<16xf32>) {
      ^bb0(%in: f16, %out: f32):
        %a = arith.extf %in : f16 to f32
        %b = arith.mulf %a, %a : f32
        %c = arith.addf %out, %b : f32
        linalg.yield %c : f32
      } -> tensor<16xf32>
      %_oi = tensor.empty() : tensor<16x128xf16>
      %_on = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%q_flat, %_ss, %w_qn : tensor<16x128xf16>, tensor<16xf32>, tensor<128xf16>) outs(%_oi : tensor<16x128xf16>) {
      ^bb0(%in: f16, %ss: f32, %w: f16, %out: f16):
        %a = arith.divf %ss, %_hdf : f32
        %b = arith.addf %a, %eps : f32
        %c = math.sqrt %b : f32
        %d = arith.extf %in : f16 to f32
        %e = arith.extf %w : f16 to f32
        %f = arith.divf %d, %c : f32
        %g = arith.mulf %f, %e : f32
        %h = arith.truncf %g : f32 to f16
        linalg.yield %h : f16
      } -> tensor<16x128xf16>
      flow.return %_on : tensor<16x128xf16>
    }
    // K norm: [8, 128]
    %k_normed = flow.dispatch.region[] -> (tensor<8x128xf16>) {
      %_cst0 = arith.constant 0.000000e+00 : f32
      %_hdf = arith.constant 128.0 : f32
      %_si = tensor.empty() : tensor<8xf32>
      %_sz = linalg.fill ins(%_cst0 : f32) outs(%_si : tensor<8xf32>) -> tensor<8xf32>
      %_ss = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%k_flat : tensor<8x128xf16>) outs(%_sz : tensor<8xf32>) {
      ^bb0(%in: f16, %out: f32):
        %a = arith.extf %in : f16 to f32
        %b = arith.mulf %a, %a : f32
        %c = arith.addf %out, %b : f32
        linalg.yield %c : f32
      } -> tensor<8xf32>
      %_oi = tensor.empty() : tensor<8x128xf16>
      %_on = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%k_flat, %_ss, %w_kn : tensor<8x128xf16>, tensor<8xf32>, tensor<128xf16>) outs(%_oi : tensor<8x128xf16>) {
      ^bb0(%in: f16, %ss: f32, %w: f16, %out: f16):
        %a = arith.divf %ss, %_hdf : f32
        %b = arith.addf %a, %eps : f32
        %c = math.sqrt %b : f32
        %d = arith.extf %in : f16 to f32
        %e = arith.extf %w : f16 to f32
        %f = arith.divf %d, %c : f32
        %g = arith.mulf %f, %e : f32
        %h = arith.truncf %g : f32 to f16
        linalg.yield %h : f16
      } -> tensor<8x128xf16>
      flow.return %_on : tensor<8x128xf16>
    }
    // Reshape for RoPE: [16, 128] -> [1, 1, 16, 128], [8, 128] -> [1, 1, 8, 128]
    %q_4d = tensor.expand_shape %q_normed [[0, 1, 2], [3]] output_shape [1, 1, 16, 128] : tensor<16x128xf16> into tensor<1x1x16x128xf16>
    %k_4d = tensor.expand_shape %k_normed [[0, 1, 2], [3]] output_shape [1, 1, 8, 128] : tensor<8x128xf16> into tensor<1x1x8x128xf16>
    // Apply RoPE
    %q_roped = util.call @rope_decode_q(%q_4d, %arg1) : (tensor<1x1x16x128xf16>, tensor<1x1xi64>) -> tensor<1x1x16x128xf16>
    %k_roped = util.call @rope_decode_k(%k_4d, %arg1) : (tensor<1x1x8x128xf16>, tensor<1x1xi64>) -> tensor<1x1x8x128xf16>
    // Concat K: cached [1, ctx_len, 8, 128] ++ new [1, 1, 8, 128] -> [1, ctx_len+1, 8, 128]
    %ctx_plus_1 = arith.addi %dim_0, %c1 : index
    // k_roped is [1,1,16,128] for Q, but we need the K version which is [1,1,8,128]
    // Reshape k_roped [1,1,8,128] -> [1,1,8,128] (already right shape)
    // For V: v_3d [1, 8, 128] -> [1, 1, 8, 128]
    %v_4d = tensor.expand_shape %v_3d [[0, 1], [2], [3]] output_shape [1, 1, 8, 128] : tensor<1x8x128xf16> into tensor<1x1x8x128xf16>
    %k_out = tensor.empty(%ctx_plus_1) : tensor<1x?x8x128xf16>
    %k_ins0 = tensor.insert_slice %arg2 into %k_out[0, 0, 0, 0] [1, %dim_0, 8, 128] [1, 1, 1, 1] : tensor<1x?x8x128xf16> into tensor<1x?x8x128xf16>
    %k_concat = tensor.insert_slice %k_roped into %k_ins0[0, %dim_0, 0, 0] [1, 1, 8, 128] [1, 1, 1, 1] : tensor<1x1x8x128xf16> into tensor<1x?x8x128xf16>
    %v_out = tensor.empty(%ctx_plus_1) : tensor<1x?x8x128xf16>
    %v_ins0 = tensor.insert_slice %arg3 into %v_out[0, 0, 0, 0] [1, %dim_0, 8, 128] [1, 1, 1, 1] : tensor<1x?x8x128xf16> into tensor<1x?x8x128xf16>
    %v_concat = tensor.insert_slice %v_4d into %v_ins0[0, %dim_0, 0, 0] [1, 1, 8, 128] [1, 1, 1, 1] : tensor<1x1x8x128xf16> into tensor<1x?x8x128xf16>
    // Attention: scale = 1/sqrt(128) = rsqrt(128)
    %head_dim_f32 = arith.constant 128.0 : f32
    %scale = math.rsqrt %head_dim_f32 : f32
    %attn_out = util.call @attention_gqa_decode_static(%q_roped, %k_concat, %v_concat, %scale) : (tensor<1x1x16x128xf16>, tensor<1x?x8x128xf16>, tensor<1x?x8x128xf16>, f32) -> tensor<1x1x16x128xf16>
    // Reshape: [1, 1, 16, 128] -> [1, 2048]
    %attn_flat = tensor.collapse_shape %attn_out [[0, 1], [2, 3]] : tensor<1x1x16x128xf16> into tensor<1x2048xf16>
    // Output fused Q8_0 dequant + transposed GEMV: [1024, 2048] @ [2048, 1] -> [1024, 1] -> [1, 1024]
    %o_xt_init = tensor.empty() : tensor<2048x1xf16>
    %o_xt = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%attn_flat : tensor<1x2048xf16>) outs(%o_xt_init : tensor<2048x1xf16>) {
    ^bb0(%in: f16, %out: f16): linalg.yield %in : f16
    } -> tensor<2048x1xf16>
    %o_xt_dyn = tensor.cast %o_xt : tensor<2048x1xf16> to tensor<?x1xf16>
    %q8_o_dyn = tensor.cast %q8_o : tensor<28x2228224xi8> to tensor<28x?xi8>
    %c1024_o = arith.constant 1024 : index
    %c2048_o = arith.constant 2048 : index
    %o_mm_dyn = util.call @q8_fused_gemv(%q8_o_dyn, %layer_i32, %o_xt_dyn, %c1024_o, %c2048_o) : (tensor<28x?xi8>, i32, tensor<?x1xf16>, index, index) -> tensor<?x1xf16>
    %o_mm = tensor.cast %o_mm_dyn : tensor<?x1xf16> to tensor<1024x1xf16>
    %o_proj_flat = tensor.collapse_shape %o_mm [[0, 1]] : tensor<1024x1xf16> into tensor<1024xf16>
    %o_proj = tensor.expand_shape %o_proj_flat [[0, 1]] output_shape [1, 1024] : tensor<1024xf16> into tensor<1x1024xf16>
    // New K for cache: collapse k_roped [1,1,8,128] -> [8, 128]
    %new_k = tensor.collapse_shape %k_roped [[0, 1, 2], [3]] : tensor<1x1x8x128xf16> into tensor<8x128xf16>
    // New V for cache: collapse v_4d [1,1,8,128] -> [8, 128]
    %new_v = tensor.collapse_shape %v_4d [[0, 1, 2], [3]] : tensor<1x1x8x128xf16> into tensor<8x128xf16>
    util.return %o_proj, %new_k, %new_v : tensor<1x1024xf16>, tensor<8x128xf16>, tensor<8x128xf16>
  }
  // ---- Static transformer layer decode ----
  // Static decode layer — takes K/V cache tensors directly (no list/import/export per layer).
  util.func private @transformer_layer_decode_static(
      %hidden: tensor<1x1024xf16>,
      %positions: tensor<1xi64>,
      %k_cache: tensor<?x8x128xf16>,
      %v_cache: tensor<?x8x128xf16>,
      %max_seq_len: index,
      %ctx_len: index,
      %layer_i32: i32
  ) -> (tensor<1x1024xf16>, tensor<?x8x128xf16>, tensor<?x8x128xf16>) {
    %c0 = arith.constant 0 : index
    %eps = arith.constant 9.99999997E-7 : f32
    // Load norm weights (f16, small)
    %attn_norm_dyn = util.call @model_params.attn_norm_weight(%layer_i32) : (i32) -> tensor<?xf16>
    %attn_norm = tensor.cast %attn_norm_dyn : tensor<?xf16> to tensor<1024xf16>
    %ffn_norm_dyn = util.call @model_params.ffn_norm_weight(%layer_i32) : (i32) -> tensor<?xf16>
    %ffn_norm = tensor.cast %ffn_norm_dyn : tensor<?xf16> to tensor<1024xf16>
    %qn_w_dyn = util.call @model_params.attn_q_norm_weight(%layer_i32) : (i32) -> tensor<?xf16>
    %qn_w = tensor.cast %qn_w_dyn : tensor<?xf16> to tensor<128xf16>
    %kn_w_dyn = util.call @model_params.attn_k_norm_weight(%layer_i32) : (i32) -> tensor<?xf16>
    %kn_w = tensor.cast %kn_w_dyn : tensor<?xf16> to tensor<128xf16>
    // Load raw Q8_0 stacked tensors (fused with GEMV below via flow.dispatch.region)
    %q8_q = flow.tensor.constant #flow.parameter.named<"model"::"stacked.q8.attn_q.weight"> : tensor<28x2228224xi8>
    %q8_k = flow.tensor.constant #flow.parameter.named<"model"::"stacked.q8.attn_k.weight"> : tensor<28x1114112xi8>
    %q8_v = flow.tensor.constant #flow.parameter.named<"model"::"stacked.q8.attn_v.weight"> : tensor<28x1114112xi8>
    %q8_o = flow.tensor.constant #flow.parameter.named<"model"::"stacked.q8.attn_output.weight"> : tensor<28x2228224xi8>
    %q8_gu = flow.tensor.constant #flow.parameter.named<"model"::"stacked.q8.ffn_gate_up.weight"> : tensor<28x6684672xi8>
    %q8_dn = flow.tensor.constant #flow.parameter.named<"model"::"stacked.q8.ffn_down.weight"> : tensor<28x3342336xi8>
    // Attention norm
    %normed = util.call @rms_norm_1x1024(%hidden, %attn_norm, %eps) : (tensor<1x1024xf16>, tensor<1024xf16>, f32) -> tensor<1x1024xf16>
    // Cache read — direct tensor slice, no import/export
    %layer_idx = arith.index_cast %layer_i32 : i32 to index
    %slot_start = arith.muli %layer_idx, %max_seq_len : index
    %k_slice = tensor.extract_slice %k_cache[%slot_start, 0, 0] [%ctx_len, 8, 128] [1, 1, 1] : tensor<?x8x128xf16> to tensor<?x8x128xf16>
    %v_slice = tensor.extract_slice %v_cache[%slot_start, 0, 0] [%ctx_len, 8, 128] [1, 1, 1] : tensor<?x8x128xf16> to tensor<?x8x128xf16>
    %c1 = arith.constant 1 : index
    %cached_k = tensor.expand_shape %k_slice [[0, 1], [2], [3]] output_shape [1, %ctx_len, 8, 128] : tensor<?x8x128xf16> into tensor<1x?x8x128xf16>
    %cached_v = tensor.expand_shape %v_slice [[0, 1], [2], [3]] output_shape [1, %ctx_len, 8, 128] : tensor<?x8x128xf16> into tensor<1x?x8x128xf16>
    // Positions: [1] -> [1, 1] for attention block
    %pos_2d = tensor.expand_shape %positions [[0, 1]] output_shape [1, 1] : tensor<1xi64> into tensor<1x1xi64>
    // Attention block
    %attn:3 = util.call @attention_block_decode_static(%normed, %pos_2d, %cached_k, %cached_v, %q8_q, %q8_k, %q8_v, %q8_o, %qn_w, %kn_w, %layer_i32) : (tensor<1x1024xf16>, tensor<1x1xi64>, tensor<1x?x8x128xf16>, tensor<1x?x8x128xf16>, tensor<28x2228224xi8>, tensor<28x1114112xi8>, tensor<28x1114112xi8>, tensor<28x2228224xi8>, tensor<128xf16>, tensor<128xf16>, i32) -> (tensor<1x1024xf16>, tensor<8x128xf16>, tensor<8x128xf16>)
    // Cache write — direct tensor insert, no import/export
    %pos_i64 = tensor.extract %positions[%c0] : tensor<1xi64>
    %pos_idx = arith.index_cast %pos_i64 : i64 to index
    %write_slot = arith.addi %slot_start, %pos_idx : index
    %k_updated = tensor.insert_slice %attn#1 into %k_cache[%write_slot, 0, 0] [1, 8, 128] [1, 1, 1] : tensor<8x128xf16> into tensor<?x8x128xf16>
    %v_updated = tensor.insert_slice %attn#2 into %v_cache[%write_slot, 0, 0] [1, 8, 128] [1, 1, 1] : tensor<8x128xf16> into tensor<?x8x128xf16>
    // Residual 1
    %cst = arith.constant 0.000000e+00 : f16
    %res1_init = tensor.empty() : tensor<1x1024xf16>
    %res1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%hidden, %attn#0 : tensor<1x1024xf16>, tensor<1x1024xf16>) outs(%res1_init : tensor<1x1024xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %r = arith.addf %in, %in_1 : f16
      linalg.yield %r : f16
    } -> tensor<1x1024xf16>
    // FFN norm
    %ffn_normed = util.call @rms_norm_1x1024(%res1, %ffn_norm, %eps) : (tensor<1x1024xf16>, tensor<1024xf16>, f32) -> tensor<1x1024xf16>
    // Gate+Up fused Q8_0 dequant + transposed GEMV: [6144, 1024] @ [1024, 1]
    %gu_xt_init = tensor.empty() : tensor<1024x1xf16>
    %gu_xt = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%ffn_normed : tensor<1x1024xf16>) outs(%gu_xt_init : tensor<1024x1xf16>) {
    ^bb0(%in: f16, %out: f16): linalg.yield %in : f16
    } -> tensor<1024x1xf16>
    %gu_xt_dyn = tensor.cast %gu_xt : tensor<1024x1xf16> to tensor<?x1xf16>
    %q8_gu_dyn = tensor.cast %q8_gu : tensor<28x6684672xi8> to tensor<28x?xi8>
    %c6144_gu = arith.constant 6144 : index
    %c1024_gu = arith.constant 1024 : index
    %gu_out_dyn = util.call @q8_fused_gemv(%q8_gu_dyn, %layer_i32, %gu_xt_dyn, %c6144_gu, %c1024_gu) : (tensor<28x?xi8>, i32, tensor<?x1xf16>, index, index) -> tensor<?x1xf16>
    %gu_out = tensor.cast %gu_out_dyn : tensor<?x1xf16> to tensor<6144x1xf16>
    %gu_flat = tensor.collapse_shape %gu_out [[0, 1]] : tensor<6144x1xf16> into tensor<6144xf16>
    %gate_up = tensor.expand_shape %gu_flat [[0, 1]] output_shape [1, 6144] : tensor<6144xf16> into tensor<1x6144xf16>
    // Split: gate [1, 3072] and up [1, 3072]
    %gate_slice = tensor.extract_slice %gate_up[0, 0] [1, 3072] [1, 1] : tensor<1x6144xf16> to tensor<1x3072xf16>
    %up_slice = tensor.extract_slice %gate_up[0, 3072] [1, 3072] [1, 1] : tensor<1x6144xf16> to tensor<1x3072xf16>
    // SwiGLU
    %swiglu_init = tensor.empty() : tensor<1x3072xf16>
    %swiglu = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%gate_slice, %up_slice : tensor<1x3072xf16>, tensor<1x3072xf16>) outs(%swiglu_init : tensor<1x3072xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %neg = arith.negf %in : f16
      %exp_neg = math.exp %neg : f16
      %one = arith.constant 1.000000e+00 : f16
      %denom = arith.addf %one, %exp_neg : f16
      %sigmoid = arith.divf %one, %denom : f16
      %silu = arith.mulf %in, %sigmoid : f16
      %result = arith.mulf %silu, %in_1 : f16
      linalg.yield %result : f16
    } -> tensor<1x3072xf16>
    // Down transposed GEMV: [1024, 3072] @ [3072, 1] -> [1024, 1] -> [1, 1024]
    %down_xt_init = tensor.empty() : tensor<3072x1xf16>
    %down_xt = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%swiglu : tensor<1x3072xf16>) outs(%down_xt_init : tensor<3072x1xf16>) {
    ^bb0(%in: f16, %out: f16): linalg.yield %in : f16
    } -> tensor<3072x1xf16>
    %dn_xt_dyn = tensor.cast %down_xt : tensor<3072x1xf16> to tensor<?x1xf16>
    %q8_dn_dyn = tensor.cast %q8_dn : tensor<28x3342336xi8> to tensor<28x?xi8>
    %c1024_dn = arith.constant 1024 : index
    %c3072_dn = arith.constant 3072 : index
    %dn_out_dyn = util.call @q8_fused_gemv(%q8_dn_dyn, %layer_i32, %dn_xt_dyn, %c1024_dn, %c3072_dn) : (tensor<28x?xi8>, i32, tensor<?x1xf16>, index, index) -> tensor<?x1xf16>
    %down_out = tensor.cast %dn_out_dyn : tensor<?x1xf16> to tensor<1024x1xf16>
    %down_flat = tensor.collapse_shape %down_out [[0, 1]] : tensor<1024x1xf16> into tensor<1024xf16>
    %ffn_out = tensor.expand_shape %down_flat [[0, 1]] output_shape [1, 1024] : tensor<1024xf16> into tensor<1x1024xf16>
    // Residual 2
    %res2_init = tensor.empty() : tensor<1x1024xf16>
    %res2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%res1, %ffn_out : tensor<1x1024xf16>, tensor<1x1024xf16>) outs(%res2_init : tensor<1x1024xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %r = arith.addf %in, %in_1 : f16
      linalg.yield %r : f16
    } -> tensor<1x1024xf16>
    util.return %res2, %k_updated, %v_updated : tensor<1x1024xf16>, tensor<?x8x128xf16>, tensor<?x8x128xf16>
  }
  // ---- Dynamic attention block (kept for prefill path's @prefill_all -> @decode fallback) ----
  util.func private @attention_block_decode_qwen_components.attention_block_decode_qwen(%arg0: tensor<?x?xf16>, %arg1: tensor<?xi64>, %arg2: tensor<?x?x?x?xf16>, %arg3: tensor<?x?x?x?xf16>, %arg4: tensor<?x?xf16>, %arg5: tensor<?x?xf16>, %arg6: tensor<?x?xf16>, %arg7: tensor<?x?xf16>, %arg8: index, %arg9: index, %arg10: index, %arg11: f32, %arg12: f32, %arg13: tensor<?xf16>, %arg14: tensor<?xf16>, %arg15: f32) -> (tensor<?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf16>
    %dim_0 = tensor.dim %arg2, %c1 : tensor<?x?x?x?xf16>
    %dim_1 = tensor.dim %arg4, %c1 : tensor<?x?xf16>
    %0 = arith.divui %dim_1, %arg8 : index
    %1 = arith.muli %arg9, %0 : index
    %c1_2 = arith.constant 1 : index
    // dyn_1: opaque "1" to prevent O1 from inserting illegal tensor.cast
    // on the concat path (tensor<?x1x?x?xf16> -> tensor<?x?x?x?xf16>).
    %dyn_1 = util.optimization_barrier %c1_2 : index
    %expanded = tensor.expand_shape %arg0 [[0], [1, 2]] output_shape [%dim, %c1_2, %arg10] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %expanded_3 = tensor.expand_shape %arg1 [[0, 1]] output_shape [%dim, %c1_2] : tensor<?xi64> into tensor<?x?xi64>
    %cst = arith.constant 0.000000e+00 : f16
    %2 = tensor.empty(%dim, %dim_1) : tensor<?x?xf16>
    %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %4 = linalg.matmul ins(%arg0, %arg4 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %5 = tensor.empty(%dim, %1) : tensor<?x?xf16>
    %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %7 = linalg.matmul ins(%arg0, %arg5 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%6 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %8 = tensor.empty(%dim, %1) : tensor<?x?xf16>
    %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %10 = linalg.matmul ins(%arg0, %arg6 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%9 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %expanded_4 = tensor.expand_shape %4 [[0], [1, 2]] output_shape [%dim, %arg8, %0] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %expanded_5 = tensor.expand_shape %7 [[0], [1, 2]] output_shape [%dim, %arg9, %0] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %expanded_6 = tensor.expand_shape %10 [[0], [1, 2]] output_shape [%dim, %arg9, %0] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %collapsed = tensor.collapse_shape %expanded_4 [[0, 1], [2]] : tensor<?x?x?xf16> into tensor<?x?xf16>
    %collapsed_7 = tensor.collapse_shape %expanded_5 [[0, 1], [2]] : tensor<?x?x?xf16> into tensor<?x?xf16>
    // Inline fused QK norm via flow.dispatch.region (forces reduction+elementwise into 1 dispatch)
    %q_bh = arith.muli %dim, %arg8 : index
    %11 = flow.dispatch.region[] -> (tensor<?x?xf16>{%q_bh, %0}) {
      %_cst0 = arith.constant 0.000000e+00 : f32
      %_hdi = arith.index_cast %0 : index to i32
      %_hdf = arith.sitofp %_hdi : i32 to f32
      %_si = tensor.empty(%q_bh) : tensor<?xf32>
      %_sz = linalg.fill ins(%_cst0 : f32) outs(%_si : tensor<?xf32>) -> tensor<?xf32>
      %_ss = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%collapsed : tensor<?x?xf16>) outs(%_sz : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %a = arith.extf %in : f16 to f32
        %b = arith.mulf %a, %a : f32
        %c = arith.addf %out, %b : f32
        linalg.yield %c : f32
      } -> tensor<?xf32>
      %_oi = tensor.empty(%q_bh, %0) : tensor<?x?xf16>
      %_on = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed, %_ss, %arg13 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>) outs(%_oi : tensor<?x?xf16>) {
      ^bb0(%in: f16, %ss: f32, %w: f16, %out: f16):
        %a = arith.divf %ss, %_hdf : f32
        %b = arith.addf %a, %arg15 : f32
        %c = math.sqrt %b : f32
        %d = arith.extf %in : f16 to f32
        %e = arith.extf %w : f16 to f32
        %f = arith.divf %d, %c : f32
        %g = arith.mulf %f, %e : f32
        %h = arith.truncf %g : f32 to f16
        linalg.yield %h : f16
      } -> tensor<?x?xf16>
      flow.return %_on : tensor<?x?xf16>
    }
    %k_bh = arith.muli %dim, %arg9 : index
    %12 = flow.dispatch.region[] -> (tensor<?x?xf16>{%k_bh, %0}) {
      %_cst0 = arith.constant 0.000000e+00 : f32
      %_hdi = arith.index_cast %0 : index to i32
      %_hdf = arith.sitofp %_hdi : i32 to f32
      %_si = tensor.empty(%k_bh) : tensor<?xf32>
      %_sz = linalg.fill ins(%_cst0 : f32) outs(%_si : tensor<?xf32>) -> tensor<?xf32>
      %_ss = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%collapsed_7 : tensor<?x?xf16>) outs(%_sz : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %a = arith.extf %in : f16 to f32
        %b = arith.mulf %a, %a : f32
        %c = arith.addf %out, %b : f32
        linalg.yield %c : f32
      } -> tensor<?xf32>
      %_oi = tensor.empty(%k_bh, %0) : tensor<?x?xf16>
      %_on = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%collapsed_7, %_ss, %arg14 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>) outs(%_oi : tensor<?x?xf16>) {
      ^bb0(%in: f16, %ss: f32, %w: f16, %out: f16):
        %a = arith.divf %ss, %_hdf : f32
        %b = arith.addf %a, %arg15 : f32
        %c = math.sqrt %b : f32
        %d = arith.extf %in : f16 to f32
        %e = arith.extf %w : f16 to f32
        %f = arith.divf %d, %c : f32
        %g = arith.mulf %f, %e : f32
        %h = arith.truncf %g : f32 to f16
        linalg.yield %h : f16
      } -> tensor<?x?xf16>
      flow.return %_on : tensor<?x?xf16>
    }
    %expanded_8 = tensor.expand_shape %11 [[0, 1], [2]] output_shape [%dim, %arg8, %0] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %expanded_9 = tensor.expand_shape %expanded_8 [[0, 1], [2], [3]] output_shape [%dim, %dyn_1, %arg8, %0] : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %expanded_10 = tensor.expand_shape %12 [[0, 1], [2]] output_shape [%dim, %arg9, %0] : tensor<?x?xf16> into tensor<?x?x?xf16>
    %expanded_11 = tensor.expand_shape %expanded_10 [[0, 1], [2], [3]] output_shape [%dim, %dyn_1, %arg9, %0] : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %cast = tensor.cast %expanded_6 : tensor<?x?x?xf16> to tensor<?x?x?xf16>
    %13 = util.call @position_components.rope(%expanded_9, %expanded_3, %arg11, %arg12) : (tensor<?x?x?x?xf16>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf16>
    %14 = util.call @position_components.rope(%expanded_11, %expanded_3, %arg11, %arg12) : (tensor<?x?x?x?xf16>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf16>
    %15 = arith.addi %dim_0, %c1 : index
    // Replace tensor.concat with insert_slice to avoid O1 tensor.cast legality issue.
    // Read actual dims from the cache tensors to avoid head-count mismatches.
    %k_dim2 = tensor.dim %arg2, %c2 : tensor<?x?x?x?xf16>
    %k_dim3 = tensor.dim %arg2, %c3 : tensor<?x?x?x?xf16>
    %v_dim2 = tensor.dim %arg3, %c2 : tensor<?x?x?x?xf16>
    %v_dim3 = tensor.dim %arg3, %c3 : tensor<?x?x?x?xf16>
    // concat K: [batch, ctx, kv_heads, head_dim] ++ [batch, 1, kv_heads, head_dim]
    %k_out = tensor.empty(%dim, %15, %k_dim2, %k_dim3) : tensor<?x?x?x?xf16>
    %k_ins0 = tensor.insert_slice %arg2 into %k_out[0, 0, 0, 0] [%dim, %dim_0, %k_dim2, %k_dim3] [1, 1, 1, 1] : tensor<?x?x?x?xf16> into tensor<?x?x?x?xf16>
    %concat = tensor.insert_slice %14 into %k_ins0[0, %dim_0, 0, 0] [%dim, %dyn_1, %k_dim2, %k_dim3] [1, 1, 1, 1] : tensor<?x?x?x?xf16> into tensor<?x?x?x?xf16>
    // concat V: same pattern
    %expanded_12 = tensor.expand_shape %cast [[0, 1], [2], [3]] output_shape [%dim, %dyn_1, %v_dim2, %v_dim3] : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %v_out = tensor.empty(%dim, %15, %v_dim2, %v_dim3) : tensor<?x?x?x?xf16>
    %v_ins0 = tensor.insert_slice %arg3 into %v_out[0, 0, 0, 0] [%dim, %dim_0, %v_dim2, %v_dim3] [1, 1, 1, 1] : tensor<?x?x?x?xf16> into tensor<?x?x?x?xf16>
    %concat_13 = tensor.insert_slice %expanded_12 into %v_ins0[0, %dim_0, 0, 0] [%dim, %dyn_1, %v_dim2, %v_dim3] [1, 1, 1, 1] : tensor<?x?x?x?xf16> into tensor<?x?x?x?xf16>
    %16 = arith.index_cast %0 : index to i32
    %17 = arith.sitofp %16 : i32 to f32
    %18 = math.rsqrt %17 : f32
    %19 = util.call @attention_components.attention_gqa(%13, %concat, %concat_13, %18) : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, f32) -> tensor<?x?x?x?xf16>
    %collapsed_14 = tensor.collapse_shape %19 [[0], [1, 2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %collapsed_15 = tensor.collapse_shape %collapsed_14 [[0], [1, 2]] : tensor<?x?x?xf16> into tensor<?x?xf16>
    %20 = tensor.empty(%dim, %arg10) : tensor<?x?xf16>
    %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %22 = linalg.matmul ins(%collapsed_15, %arg7 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%21 : tensor<?x?xf16>) -> tensor<?x?xf16>
    %collapsed_16 = tensor.collapse_shape %14 [[0], [1, 2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %collapsed_17 = tensor.collapse_shape %expanded_12 [[0], [1, 2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    util.return %22, %collapsed_16, %collapsed_17 : tensor<?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>
  }
  // ---- Dynamic GQA attention (kept for prefill path) ----
  util.func private @attention_components.attention_gqa(%arg0: tensor<?x?x?x?xf16>, %arg1: tensor<?x?x?x?xf16>, %arg2: tensor<?x?x?x?xf16>, %arg3: f32) -> tensor<?x?x?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x?x?xf16>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf16>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf16>
    %dim_3 = tensor.dim %arg1, %c2 : tensor<?x?x?x?xf16>
    %dim_4 = tensor.dim %arg1, %c1 : tensor<?x?x?x?xf16>
    %0 = tensor.empty(%dim, %dim_1, %dim_0, %dim_2) : tensor<?x?x?x?xf16>
    %1 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x?x?x?xf16>) outs(%0 : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>
    %2 = tensor.empty(%dim, %dim_3, %dim_4, %dim_2) : tensor<?x?x?x?xf16>
    %3 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<?x?x?x?xf16>) outs(%2 : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>
    %4 = tensor.empty(%dim, %dim_3, %dim_4, %dim_2) : tensor<?x?x?x?xf16>
    %5 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<?x?x?x?xf16>) outs(%4 : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>
    %6 = arith.divui %dim_1, %dim_3 : index
    %7 = tensor.empty(%dim, %dim_3, %6, %dim_4, %dim_2) : tensor<?x?x?x?x?xf16>
    %8 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<?x?x?x?xf16>) outs(%7 : tensor<?x?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?x?xf16>
    %collapsed = tensor.collapse_shape %8 [[0], [1, 2], [3], [4]] : tensor<?x?x?x?x?xf16> into tensor<?x?x?x?xf16>
    %9 = tensor.empty(%dim, %dim_3, %6, %dim_4, %dim_2) : tensor<?x?x?x?x?xf16>
    %10 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%5 : tensor<?x?x?x?xf16>) outs(%9 : tensor<?x?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?x?xf16>
    %collapsed_5 = tensor.collapse_shape %10 [[0], [1, 2], [3], [4]] : tensor<?x?x?x?x?xf16> into tensor<?x?x?x?xf16>
    %11 = arith.muli %dim, %dim_1 : index
    %collapsed_6 = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %collapsed_7 = tensor.collapse_shape %collapsed [[0, 1], [2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %collapsed_8 = tensor.collapse_shape %collapsed_5 [[0, 1], [2], [3]] : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %12 = arith.subi %dim_4, %dim_0 : index
    %13 = tensor.empty(%11, %dim_0, %dim_4) : tensor<?x?x?xi1>
    %14 = linalg.generic {indexing_maps = [#map8], iterator_types = ["parallel", "parallel", "parallel"]} outs(%13 : tensor<?x?x?xi1>) {
    ^bb0(%out: i1):
      %19 = linalg.index 1 : index
      %20 = linalg.index 2 : index
      %21 = arith.addi %19, %12 : index
      %22 = arith.cmpi ule, %20, %21 : index
      linalg.yield %22 : i1
    } -> tensor<?x?x?xi1>
    %15 = tensor.empty(%11, %dim_0, %dim_2) : tensor<?x?x?xf16>
    %16 = iree_linalg_ext.attention {indexing_maps = [#map9, #map10, #map11, #map12, #map13, #map14]} ins(%collapsed_6, %collapsed_7, %collapsed_8, %arg3, %14 : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, f32, tensor<?x?x?xi1>) outs(%15 : tensor<?x?x?xf16>) {
    ^bb0(%arg4: f32):
      iree_linalg_ext.yield %arg4 : f32
    } -> tensor<?x?x?xf16>
    %expanded = tensor.expand_shape %16 [[0, 1], [2], [3]] output_shape [%dim, %dim_1, %dim_0, %dim_2] : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %17 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf16>
    %18 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<?x?x?x?xf16>) outs(%17 : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>
    util.return %18 : tensor<?x?x?x?xf16>
  }
  // ---- Dynamic RoPE (kept for prefill path) ----
  util.func private @position_components.rope(%arg0: tensor<?x?x?x?xf16>, %arg1: tensor<?x?xi64>, %arg2: f32, %arg3: f32) -> tensor<?x?x?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?x?xf16>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x?x?xf16>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf16>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf16>
    %0 = arith.divsi %dim_2, %c2 : index
    %1 = arith.index_cast %dim_2 : index to i32
    %2 = arith.sitofp %1 : i32 to f32
    %3 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf16>
    %4 = linalg.generic {indexing_maps = [#map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3 : tensor<?x?x?x?xf16>) {
    ^bb0(%out: f16):
      %5 = linalg.index 0 : index
      %6 = linalg.index 1 : index
      %7 = linalg.index 2 : index
      %8 = linalg.index 3 : index
      %extracted = tensor.extract %arg1[%5, %6] : tensor<?x?xi64>
      %9 = arith.trunci %extracted : i64 to i32
      %10 = arith.sitofp %9 : i32 to f32
      %11 = arith.cmpi slt, %8, %0 : index
      %12 = scf.if %11 -> (f16) {
        %13 = arith.index_cast %8 : index to i32
        %14 = arith.sitofp %13 : i32 to f32
        %cst = arith.constant 2.000000e+00 : f32
        %15 = arith.mulf %cst, %14 : f32
        %16 = arith.divf %15, %2 : f32
        %17 = arith.negf %16 : f32
        %18 = math.powf %arg2, %17 : f32
        %19 = arith.mulf %18, %arg3 : f32
        %20 = arith.mulf %10, %19 : f32
        %21 = math.cos %20 : f32
        %22 = math.sin %20 : f32
        %extracted_3 = tensor.extract %arg0[%5, %6, %7, %8] : tensor<?x?x?x?xf16>
        %23 = arith.addi %8, %0 : index
        %extracted_4 = tensor.extract %arg0[%5, %6, %7, %23] : tensor<?x?x?x?xf16>
        %24 = arith.extf %extracted_3 : f16 to f32
        %25 = arith.extf %extracted_4 : f16 to f32
        %26 = arith.mulf %24, %21 : f32
        %27 = arith.mulf %25, %22 : f32
        %28 = arith.subf %26, %27 : f32
        %29 = arith.truncf %28 : f32 to f16
        scf.yield %29 : f16
      } else {
        %13 = arith.subi %8, %0 : index
        %14 = arith.index_cast %13 : index to i32
        %15 = arith.sitofp %14 : i32 to f32
        %cst = arith.constant 2.000000e+00 : f32
        %16 = arith.mulf %cst, %15 : f32
        %17 = arith.divf %16, %2 : f32
        %18 = arith.negf %17 : f32
        %19 = math.powf %arg2, %18 : f32
        %20 = arith.mulf %19, %arg3 : f32
        %21 = arith.mulf %10, %20 : f32
        %22 = math.cos %21 : f32
        %23 = math.sin %21 : f32
        %extracted_3 = tensor.extract %arg0[%5, %6, %7, %8] : tensor<?x?x?x?xf16>
        %extracted_4 = tensor.extract %arg0[%5, %6, %7, %13] : tensor<?x?x?x?xf16>
        %24 = arith.extf %extracted_4 : f16 to f32
        %25 = arith.extf %extracted_3 : f16 to f32
        %26 = arith.mulf %24, %23 : f32
        %27 = arith.mulf %25, %22 : f32
        %28 = arith.addf %26, %27 : f32
        %29 = arith.truncf %28 : f32 to f16
        scf.yield %29 : f16
      }
      linalg.yield %12 : f16
    } -> tensor<?x?x?x?xf16>
    util.return %4 : tensor<?x?x?x?xf16>
  }
  util.func private @model_params.ffn_down_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_down.weight"> : tensor<28x3145728xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 3145728] [1, 1] : tensor<28x3145728xf16> to tensor<1x3145728xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x3145728xf16> into tensor<3145728xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [3072, 1024] : tensor<3145728xf16> into tensor<3072x1024xf16>
    %cast = tensor.cast %expanded : tensor<3072x1024xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.ffn_gate_up_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_gate_up.weight"> : tensor<28x6291456xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 6291456] [1, 1] : tensor<28x6291456xf16> to tensor<1x6291456xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x6291456xf16> into tensor<6291456xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [1024, 6144] : tensor<6291456xf16> into tensor<1024x6144xf16>
    %cast = tensor.cast %expanded : tensor<1024x6144xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.attn_k_norm_weight(%arg0: i32) -> tensor<?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_k_norm.weight"> : tensor<28x128xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 128] [1, 1] : tensor<28x128xf16> to tensor<1x128xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x128xf16> into tensor<128xf16>
    %cast = tensor.cast %collapsed : tensor<128xf16> to tensor<?xf16>
    util.return %cast : tensor<?xf16>
  }
  util.func private @model_params.attn_q_norm_weight(%arg0: i32) -> tensor<?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_q_norm.weight"> : tensor<28x128xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 128] [1, 1] : tensor<28x128xf16> to tensor<1x128xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x128xf16> into tensor<128xf16>
    %cast = tensor.cast %collapsed : tensor<128xf16> to tensor<?xf16>
    util.return %cast : tensor<?xf16>
  }
  util.func private @model_params.attn_output_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_output.weight"> : tensor<28x2097152xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 2097152] [1, 1] : tensor<28x2097152xf16> to tensor<1x2097152xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x2097152xf16> into tensor<2097152xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [2048, 1024] : tensor<2097152xf16> into tensor<2048x1024xf16>
    %cast = tensor.cast %expanded : tensor<2048x1024xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.attn_v_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_v.weight"> : tensor<28x1048576xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 1048576] [1, 1] : tensor<28x1048576xf16> to tensor<1x1048576xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x1048576xf16> into tensor<1048576xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [1024, 1024] : tensor<1048576xf16> into tensor<1024x1024xf16>
    %cast = tensor.cast %expanded : tensor<1024x1024xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.attn_k_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_k.weight"> : tensor<28x1048576xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 1048576] [1, 1] : tensor<28x1048576xf16> to tensor<1x1048576xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x1048576xf16> into tensor<1048576xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [1024, 1024] : tensor<1048576xf16> into tensor<1024x1024xf16>
    %cast = tensor.cast %expanded : tensor<1024x1024xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.attn_q_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_q.weight"> : tensor<28x2097152xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 2097152] [1, 1] : tensor<28x2097152xf16> to tensor<1x2097152xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x2097152xf16> into tensor<2097152xf16>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [1024, 2048] : tensor<2097152xf16> into tensor<1024x2048xf16>
    %cast = tensor.cast %expanded : tensor<1024x2048xf16> to tensor<?x?xf16>
    util.return %cast : tensor<?x?xf16>
  }
  util.func private @model_params.ffn_norm_weight(%arg0: i32) -> tensor<?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.ffn_norm.weight"> : tensor<28x1024xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 1024] [1, 1] : tensor<28x1024xf16> to tensor<1x1024xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x1024xf16> into tensor<1024xf16>
    %cast = tensor.cast %collapsed : tensor<1024xf16> to tensor<?xf16>
    util.return %cast : tensor<?xf16>
  }
  util.func private @model_params.attn_norm_weight(%arg0: i32) -> tensor<?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.attn_norm.weight"> : tensor<28x1024xf16>
    %1 = arith.index_cast %arg0 : i32 to index
    %extracted_slice = tensor.extract_slice %0[%1, 0] [1, 1024] [1, 1] : tensor<28x1024xf16> to tensor<1x1024xf16>
    %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x1024xf16> into tensor<1024xf16>
    %cast = tensor.cast %collapsed : tensor<1024xf16> to tensor<?xf16>
    util.return %cast : tensor<?xf16>
  }
  // ---- Q8_0 dequant: [n_blocks, 34] i8 -> [K, N] f16 ----
  // Q8_0 block: bytes[0:2] = f16 scale, bytes[2:34] = 32 × i8 values
  // dequant(block, elem) = scale * i8_val
  // Generic helper: loads from stacked Q8 param, extracts layer, dequants to f16.
  // q8_data: tensor<28 x bytes_per_layer x i8>, layer: i32
  // Returns: tensor<K x N x f16>
  util.func private @dequant_q8_layer_KxN(
      %q8_stacked: tensor<28x?xi8>,
      %layer: i32,
      %K: index, %N: index
  ) -> tensor<?x?xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2_idx = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c34 = arith.constant 34 : index
    %layer_idx = arith.index_cast %layer : i32 to index
    // bytes_per_layer
    %bpl = tensor.dim %q8_stacked, %c1 : tensor<28x?xi8>
    // Extract this layer's raw bytes
    %raw_slice = tensor.extract_slice %q8_stacked[%layer_idx, 0] [1, %bpl] [1, 1]
        : tensor<28x?xi8> to tensor<1x?xi8>
    %raw = tensor.collapse_shape %raw_slice [[0, 1]] : tensor<1x?xi8> into tensor<?xi8>
    // Dequant to [K, N] f16
    %init = tensor.empty(%K, %N) : tensor<?x?xf16>
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } outs(%init : tensor<?x?xf16>) {
    ^bb0(%out: f16):
      %k = linalg.index 0 : index
      %n = linalg.index 1 : index
      // flat_idx = k * N + n
      %k_times_N = arith.muli %k, %N : index
      %flat_idx = arith.addi %k_times_N, %n : index
      // block_idx = flat_idx / 32
      %block_idx = arith.divui %flat_idx, %c32 : index
      %elem_in_block = arith.remui %flat_idx, %c32 : index
      // byte offset in raw: block_idx * 34
      %block_byte_off = arith.muli %block_idx, %c34 : index
      // Scale: bytes [block_byte_off, block_byte_off+1] as f16
      %s1_off = arith.addi %block_byte_off, %c1 : index
      %s0 = tensor.extract %raw[%block_byte_off] : tensor<?xi8>
      %s1 = tensor.extract %raw[%s1_off] : tensor<?xi8>
      %s0_i16 = arith.extui %s0 : i8 to i16
      %s1_i16 = arith.extui %s1 : i8 to i16
      %c8_i16 = arith.constant 8 : i16
      %s1_sh = arith.shli %s1_i16, %c8_i16 : i16
      %scale_i16 = arith.ori %s0_i16, %s1_sh : i16
      %scale_f16 = arith.bitcast %scale_i16 : i16 to f16
      // Value: byte at block_byte_off + 2 + elem_in_block
      %val_base = arith.addi %block_byte_off, %c2_idx : index
      %val_off = arith.addi %val_base, %elem_in_block : index
      %qval = tensor.extract %raw[%val_off] : tensor<?xi8>
      %qval_f16 = arith.sitofp %qval : i8 to f16
      %dq = arith.mulf %scale_f16, %qval_f16 : f16
      linalg.yield %dq : f16
    } -> tensor<?x?xf16>
    util.return %result : tensor<?x?xf16>
  }
  // ---- Q8_0 param getters ----
  // Q8 getters: dequant to GGUF native [N, K] layout for transposed GEMV
  util.func private @model_params.q8_attn_q_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.q8.attn_q.weight"> : tensor<28x2228224xi8>
    %1 = tensor.cast %0 : tensor<28x2228224xi8> to tensor<28x?xi8>
    %c2048 = arith.constant 2048 : index
    %c1024 = arith.constant 1024 : index
    %r = util.call @dequant_q8_layer_KxN(%1, %arg0, %c2048, %c1024) : (tensor<28x?xi8>, i32, index, index) -> tensor<?x?xf16>
    util.return %r : tensor<?x?xf16>
  }
  util.func private @model_params.q8_attn_k_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.q8.attn_k.weight"> : tensor<28x1114112xi8>
    %1 = tensor.cast %0 : tensor<28x1114112xi8> to tensor<28x?xi8>
    %c1024 = arith.constant 1024 : index
    %r = util.call @dequant_q8_layer_KxN(%1, %arg0, %c1024, %c1024) : (tensor<28x?xi8>, i32, index, index) -> tensor<?x?xf16>
    util.return %r : tensor<?x?xf16>
  }
  util.func private @model_params.q8_attn_v_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.q8.attn_v.weight"> : tensor<28x1114112xi8>
    %1 = tensor.cast %0 : tensor<28x1114112xi8> to tensor<28x?xi8>
    %c1024 = arith.constant 1024 : index
    %r = util.call @dequant_q8_layer_KxN(%1, %arg0, %c1024, %c1024) : (tensor<28x?xi8>, i32, index, index) -> tensor<?x?xf16>
    util.return %r : tensor<?x?xf16>
  }
  util.func private @model_params.q8_attn_output_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.q8.attn_output.weight"> : tensor<28x2228224xi8>
    %1 = tensor.cast %0 : tensor<28x2228224xi8> to tensor<28x?xi8>
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    %r = util.call @dequant_q8_layer_KxN(%1, %arg0, %c1024, %c2048) : (tensor<28x?xi8>, i32, index, index) -> tensor<?x?xf16>
    util.return %r : tensor<?x?xf16>
  }
  util.func private @model_params.q8_ffn_down_weight(%arg0: i32) -> tensor<?x?xf16> {
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.q8.ffn_down.weight"> : tensor<28x3342336xi8>
    %1 = tensor.cast %0 : tensor<28x3342336xi8> to tensor<28x?xi8>
    %c1024 = arith.constant 1024 : index
    %c3072 = arith.constant 3072 : index
    %r = util.call @dequant_q8_layer_KxN(%1, %arg0, %c1024, %c3072) : (tensor<28x?xi8>, i32, index, index) -> tensor<?x?xf16>
    util.return %r : tensor<?x?xf16>
  }
  util.func private @model_params.q8_ffn_gate_up_weight(%arg0: i32) -> tensor<?x?xf16> {
    // GGUF native: gate [3072,1024] + up [3072,1024] = [6144, 1024]
    %0 = flow.tensor.constant #flow.parameter.named<"model"::"stacked.q8.ffn_gate_up.weight"> : tensor<28x6684672xi8>
    %1 = tensor.cast %0 : tensor<28x6684672xi8> to tensor<28x?xi8>
    %c6144 = arith.constant 6144 : index
    %c1024 = arith.constant 1024 : index
    %r = util.call @dequant_q8_layer_KxN(%1, %arg0, %c6144, %c1024) : (tensor<28x?xi8>, i32, index, index) -> tensor<?x?xf16>
    util.return %r : tensor<?x?xf16>
  }
  // ---- allocate_kv_cache: direct flat cache ----
  util.func public @allocate_kv_cache(%max_seq_len_val: index) -> !util.list<?> {
    %0 = util.call @hparams.attention_head_count_kv() : () -> i64
    %1 = util.call @hparams.block_count() : () -> i64
    %3 = arith.index_cast %0 : i64 to index
    %n_layers = arith.index_cast %1 : i64 to index
    %6 = util.call @hparams.head_dim() : () -> i64
    %7 = arith.index_cast %6 : i64 to index
    %total_slots = arith.muli %n_layers, %max_seq_len_val : index
    %8 = util.call @kvcache_components.allocate(%total_slots, %3, %7) : (index, index, index) -> !util.list<?>
    util.return %8 : !util.list<?>
  }
  // ---- decode: static shapes for decode path ----
  util.func public @decode(%arg0: tensor<?xi64>, %arg1: tensor<?xi64>, %arg2: !util.list<?>, %max_seq_len_val: index, %ctx_len: index) -> (tensor<?x?xf16>, !util.list<?>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c28 = arith.constant 28 : index
    %eps = arith.constant 9.99999997E-7 : f32
    // Cast inputs to static shapes
    %tokens = tensor.cast %arg0 : tensor<?xi64> to tensor<1xi64>
    %positions = tensor.cast %arg1 : tensor<?xi64> to tensor<1xi64>
    // Embedding lookup
    %embd_w_dyn = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf16>
    %embd_w = tensor.cast %embd_w_dyn : tensor<?x?xf16> to tensor<151936x1024xf16>
    %embd_dyn = util.call @embedding_components.embedding_lookup_1d(%embd_w_dyn, %arg0) : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>
    %embd = tensor.cast %embd_dyn : tensor<?x?xf16> to tensor<1x1024xf16>
    // Unpack K/V tensors from list ONCE before the layer loop
    %k_bv_pre = util.list.get %arg2[%c0] : !util.list<?> -> !hal.buffer_view
    %v_bv_pre = util.list.get %arg2[%c1] : !util.list<?> -> !hal.buffer_view
    %total_pre = hal.buffer_view.dim<%k_bv_pre : !hal.buffer_view>[0] : index
    %k_tensor_pre = hal.tensor.import %k_bv_pre : !hal.buffer_view -> tensor<?x8x128xf16>{%total_pre}
    %v_tensor_pre = hal.tensor.import %v_bv_pre : !hal.buffer_view -> tensor<?x8x128xf16>{%total_pre}
    // Layer loop — pass K/V as tensors, no import/export per layer
    %result:3 = scf.for %layer_iv = %c0 to %c28 step %c1 iter_args(%h = %embd, %kc = %k_tensor_pre, %vc = %v_tensor_pre) -> (tensor<1x1024xf16>, tensor<?x8x128xf16>, tensor<?x8x128xf16>) {
      %li32 = arith.index_cast %layer_iv : index to i32
      %layer_out:3 = util.call @transformer_layer_decode_static(%h, %positions, %kc, %vc, %max_seq_len_val, %ctx_len, %li32) : (tensor<1x1024xf16>, tensor<1xi64>, tensor<?x8x128xf16>, tensor<?x8x128xf16>, index, index, i32) -> (tensor<1x1024xf16>, tensor<?x8x128xf16>, tensor<?x8x128xf16>)
      scf.yield %layer_out#0, %layer_out#1, %layer_out#2 : tensor<1x1024xf16>, tensor<?x8x128xf16>, tensor<?x8x128xf16>
    }
    // Repack K/V tensors into list ONCE after the layer loop
    %k_bv_post = hal.tensor.export %result#1 : tensor<?x8x128xf16>{%total_pre} -> !hal.buffer_view
    %v_bv_post = hal.tensor.export %result#2 : tensor<?x8x128xf16>{%total_pre} -> !hal.buffer_view
    util.list.set %arg2[%c0], %k_bv_post : !hal.buffer_view -> !util.list<?>
    util.list.set %arg2[%c1], %v_bv_post : !hal.buffer_view -> !util.list<?>
    // Final norm
    %output_norm_dyn = util.call @model_params.output_norm_weight() : () -> tensor<?xf16>
    %output_norm = tensor.cast %output_norm_dyn : tensor<?xf16> to tensor<1024xf16>
    %normed = util.call @rms_norm_1x1024(%result#0, %output_norm, %eps) : (tensor<1x1024xf16>, tensor<1024xf16>, f32) -> tensor<1x1024xf16>
    // Output projection: transposed GEMV for 21x speedup
    // W^T @ x^T: [151936, 1024] @ [1024, 1] -> [151936, 1] -> [1, 151936]
    %output_wt = util.call @model_params.output_weight_T() : () -> tensor<151936x1024xf16>
    %cst_0 = arith.constant 0.000000e+00 : f16
    %xt_init = tensor.empty() : tensor<1024x1xf16>
    %normed_t = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%normed : tensor<1x1024xf16>) outs(%xt_init : tensor<1024x1xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<1024x1xf16>
    %logits_t_init = tensor.empty() : tensor<151936x1xf16>
    %logits_t_zero = linalg.fill ins(%cst_0 : f16) outs(%logits_t_init : tensor<151936x1xf16>) -> tensor<151936x1xf16>
    %logits_t = linalg.matmul ins(%output_wt, %normed_t : tensor<151936x1024xf16>, tensor<1024x1xf16>) outs(%logits_t_zero : tensor<151936x1xf16>) -> tensor<151936x1xf16>
    %logits_flat = tensor.collapse_shape %logits_t [[0, 1]] : tensor<151936x1xf16> into tensor<151936xf16>
    %logits = tensor.expand_shape %logits_flat [[0, 1]] output_shape [1, 151936] : tensor<151936xf16> into tensor<1x151936xf16>
    // Cast back to dynamic for external interface
    %logits_dyn = tensor.cast %logits : tensor<1x151936xf16> to tensor<?x?xf16>
    util.return %logits_dyn, %arg2 : tensor<?x?xf16>, !util.list<?>
  }
  // ---- generate: static decode loop ----
  util.func public @generate(%arg0: i64, %arg1: !util.list<?>, %max_seq_len_val: index, %arg3: index, %arg4: i64, %arg5: i64) -> (tensor<?xi64>, index, !util.list<?>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %true = arith.constant true
    %cst = arith.constant 0.000000e+00 : f16
    %c28 = arith.constant 28 : index
    %c151936 = arith.constant 151936 : index
    %eps = arith.constant 9.99999997E-7 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %embd_w_dyn = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf16>
    %output_norm_dyn = util.call @model_params.output_norm_weight() : () -> tensor<?xf16>
    %output_norm = tensor.cast %output_norm_dyn : tensor<?xf16> to tensor<1024xf16>
    %output_wt_gen = util.call @model_params.output_weight_T() : () -> tensor<151936x1024xf16>
    // Batched generation: inner scf.for of 16 tokens (no sync), outer scf.while checks EOS.
    %c_batch = arith.constant 4 : index
    %17 = tensor.empty(%arg3) : tensor<?xi64>
    %18 = linalg.fill ins(%c0_i64 : i64) outs(%17 : tensor<?xi64>) -> tensor<?xi64>
    %start_ctx = arith.index_cast %arg5 : i64 to index
    %20:6 = scf.while (%arg6 = %arg0, %arg7 = %start_ctx, %arg8 = %arg1, %arg9 = %18, %arg10 = %c0, %arg_not_eos = %true) : (i64, index, !util.list<?>, tensor<?xi64>, index, i1) -> (i64, index, !util.list<?>, tensor<?xi64>, index, i1) {
      %still_count = arith.cmpi ult, %arg10, %arg3 : index
      %still_going = arith.andi %still_count, %arg_not_eos : i1
      scf.condition(%still_going) %arg6, %arg7, %arg8, %arg9, %arg10, %arg_not_eos : i64, index, !util.list<?>, tensor<?xi64>, index, i1
    } do {
    ^bb0(%arg6: i64, %arg7: index, %arg8: !util.list<?>, %arg9: tensor<?xi64>, %arg10: index, %arg_ne: i1):
      // Compute batch size: min(16, remaining)
      %remaining = arith.subi %arg3, %arg10 : index
      %batch_size = arith.minui %remaining, %c_batch : index
      // Inner batch: generate batch_size tokens
      // Prepare initial token and position as tensors
      %init_tok_tensor = tensor.from_elements %arg6 : tensor<1xi64>
      %pos_as_i64 = arith.index_cast %arg7 : index to i64
      %init_pos_tensor = tensor.from_elements %pos_as_i64 : tensor<1xi64>
      // Offset for position increments (CPU-side counter, no GPU sync needed)
      %inner:5 = scf.for %bi = %c0 to %batch_size step %c1
          iter_args(%tok_t = %init_tok_tensor, %ctx_len_v = %arg7, %out_toks = %arg9, %out_idx = %arg10, %pos_t = %init_pos_tensor)
          -> (tensor<1xi64>, index, tensor<?xi64>, index, tensor<1xi64>) {
        // Embedding — tok_t is already tensor<1xi64>, no extract needed
        %tok_dyn = tensor.cast %tok_t : tensor<1xi64> to tensor<?xi64>
        %embd_dyn = util.call @embedding_components.embedding_lookup_1d(%embd_w_dyn, %tok_dyn) : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>
        %embd = tensor.cast %embd_dyn : tensor<?x?xf16> to tensor<1x1024xf16>
        // Unpack K/V from list
        %k_bv_g = util.list.get %arg8[%c0] : !util.list<?> -> !hal.buffer_view
        %v_bv_g = util.list.get %arg8[%c1] : !util.list<?> -> !hal.buffer_view
        %total_g = hal.buffer_view.dim<%k_bv_g : !hal.buffer_view>[0] : index
        %k_dyn = hal.tensor.import %k_bv_g : !hal.buffer_view -> tensor<?x8x128xf16>{%total_g}
        %v_dyn = hal.tensor.import %v_bv_g : !hal.buffer_view -> tensor<?x8x128xf16>{%total_g}
        // Layer loop
        %layer_result:3 = scf.for %arg12 = %c0 to %c28 step %c1 iter_args(%h = %embd, %kc = %k_dyn, %vc = %v_dyn) -> (tensor<1x1024xf16>, tensor<?x8x128xf16>, tensor<?x8x128xf16>) {
          %li32 = arith.index_cast %arg12 : index to i32
          %lo:3 = util.call @transformer_layer_decode_static(%h, %pos_t, %kc, %vc, %max_seq_len_val, %ctx_len_v, %li32) : (tensor<1x1024xf16>, tensor<1xi64>, tensor<?x8x128xf16>, tensor<?x8x128xf16>, index, index, i32) -> (tensor<1x1024xf16>, tensor<?x8x128xf16>, tensor<?x8x128xf16>)
          scf.yield %lo#0, %lo#1, %lo#2 : tensor<1x1024xf16>, tensor<?x8x128xf16>, tensor<?x8x128xf16>
        }
        // Repack K/V
        %k_bv_gp = hal.tensor.export %layer_result#1 : tensor<?x8x128xf16>{%total_g} -> !hal.buffer_view
        %v_bv_gp = hal.tensor.export %layer_result#2 : tensor<?x8x128xf16>{%total_g} -> !hal.buffer_view
        util.list.set %arg8[%c0], %k_bv_gp : !hal.buffer_view -> !util.list<?>
        util.list.set %arg8[%c1], %v_bv_gp : !hal.buffer_view -> !util.list<?>
        // Norm + transposed vocab projection
        %normed = util.call @rms_norm_1x1024(%layer_result#0, %output_norm, %eps) : (tensor<1x1024xf16>, tensor<1024xf16>, f32) -> tensor<1x1024xf16>
        %xt_init = tensor.empty() : tensor<1024x1xf16>
        %normed_t = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]
        } ins(%normed : tensor<1x1024xf16>) outs(%xt_init : tensor<1024x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          linalg.yield %in : f16
        } -> tensor<1024x1xf16>
        %lt_init = tensor.empty() : tensor<151936x1xf16>
        %lt_zero = linalg.fill ins(%cst : f16) outs(%lt_init : tensor<151936x1xf16>) -> tensor<151936x1xf16>
        %lt = linalg.matmul ins(%output_wt_gen, %normed_t : tensor<151936x1024xf16>, tensor<1024x1xf16>) outs(%lt_zero : tensor<151936x1xf16>) -> tensor<151936x1xf16>
        %lt_flat = tensor.collapse_shape %lt [[0, 1]] : tensor<151936x1xf16> into tensor<151936xf16>
        // Tiled argmax
        %cst_4 = arith.constant 0xFC00 : f16
        %c_neg1 = arith.constant -1 : i64
        %c256_am = arith.constant 256 : index
        %pad_init = tensor.empty() : tensor<152064xf16>
        %pad_fill = linalg.fill ins(%cst_4 : f16) outs(%pad_init : tensor<152064xf16>) -> tensor<152064xf16>
        %padded = tensor.insert_slice %lt_flat into %pad_fill[0] [151936] [1] : tensor<151936xf16> into tensor<152064xf16>
        %reshaped_am = tensor.expand_shape %padded [[0, 1]] output_shape [594, 256] : tensor<152064xf16> into tensor<594x256xf16>
        %v1_init = tensor.empty() : tensor<594xf16>
        %i1_init = tensor.empty() : tensor<594xi64>
        %v1_fill = linalg.fill ins(%cst_4 : f16) outs(%v1_init : tensor<594xf16>) -> tensor<594xf16>
        %i1_fill = linalg.fill ins(%c_neg1 : i64) outs(%i1_init : tensor<594xi64>) -> tensor<594xi64>
        %p1:2 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]
        } ins(%reshaped_am : tensor<594x256xf16>) outs(%v1_fill, %i1_fill : tensor<594xf16>, tensor<594xi64>) {
        ^bb0(%in: f16, %ov: f16, %oi: i64):
          %d0 = linalg.index 0 : index
          %d1 = linalg.index 1 : index
          %g = arith.muli %d0, %c256_am : index
          %g2 = arith.addi %g, %d1 : index
          %gi = arith.index_cast %g2 : index to i64
          %cmp = arith.cmpf ogt, %in, %ov : f16
          %nv = arith.select %cmp, %in, %ov : f16
          %ni = arith.select %cmp, %gi, %oi : i64
          linalg.yield %nv, %ni : f16, i64
        } -> (tensor<594xf16>, tensor<594xi64>)
        %fv_init = tensor.empty() : tensor<f16>
        %fi_init = tensor.empty() : tensor<i64>
        %fv = linalg.fill ins(%cst_4 : f16) outs(%fv_init : tensor<f16>) -> tensor<f16>
        %fi = linalg.fill ins(%c_neg1 : i64) outs(%fi_init : tensor<i64>) -> tensor<i64>
        %final:2 = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>],
          iterator_types = ["reduction"]
        } ins(%p1#0, %p1#1 : tensor<594xf16>, tensor<594xi64>) outs(%fv, %fi : tensor<f16>, tensor<i64>) {
        ^bb0(%iv: f16, %ii: i64, %ov: f16, %oi: i64):
          %cmp = arith.cmpf ogt, %iv, %ov : f16
          %nv = arith.select %cmp, %iv, %ov : f16
          %ni = arith.select %cmp, %ii, %oi : i64
          linalg.yield %nv, %ni : f16, i64
        } -> (tensor<f16>, tensor<i64>)
        %next_tok = tensor.extract %final#1[] : tensor<i64>
        %next_tok_1d = tensor.expand_shape %final#1 [] output_shape [1] : tensor<i64> into tensor<1xi64>
        %new_out = tensor.insert %next_tok into %out_toks[%out_idx] : tensor<?xi64>
        // Increment position (CPU-side) and position tensor (GPU-side)
        %next_ctx = arith.addi %ctx_len_v, %c1 : index
        %pos_inc = tensor.empty() : tensor<1xi64>
        %next_pos_t = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%pos_t : tensor<1xi64>) outs(%pos_inc : tensor<1xi64>) {
        ^bb0(%in: i64, %out: i64):
          %inc = arith.addi %in, %c1_i64 : i64
          linalg.yield %inc : i64
        } -> tensor<1xi64>
        %next_idx = arith.addi %out_idx, %c1 : index
        scf.yield %next_tok_1d, %next_ctx, %new_out, %next_idx, %next_pos_t : tensor<1xi64>, index, tensor<?xi64>, index, tensor<1xi64>
      }
      // Check EOS
      %last_tok_scalar = tensor.extract %inner#0[%c0] : tensor<1xi64>
      %is_eos = arith.cmpi eq, %last_tok_scalar, %arg4 : i64
      %not_eos = arith.xori %is_eos, %true : i1
      scf.yield %last_tok_scalar, %inner#1, %arg8, %inner#2, %inner#3, %not_eos : i64, index, !util.list<?>, tensor<?xi64>, index, i1
    }
    util.return %20#3, %20#4, %20#2 : tensor<?xi64>, index, !util.list<?>
  }
  // Prefill all tokens via scf.for calling @decode for each.
  // This is a SEPARATE function (not inlined into @run) to prevent
  // canonicalization from seeing the list.set in @allocate_kv_cache
  // and the scf.for in the same scope (which triggers dead-store removal).
  // Matches OLMo's @prefill_all structure exactly: carry logits through
  // the loop as an iter_arg, no double-decode of the last token.
  util.func public @prefill_all(%arg0: tensor<?xi64>, %arg1: index, %cache: !util.list<?>, %max_seq_len_val: index) -> (tensor<?x?xf16>, !util.list<?>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %bc = util.call @hparams.block_count() : () -> i64
    %n_layers = arith.index_cast %bc : i64 to index
    %emb_i64 = util.call @hparams.embedding_length() : () -> i64
    %emb_dim = arith.index_cast %emb_i64 : i64 to index
    // Initial dummy logits (overwritten on first iteration)
    %vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %vocab = arith.index_cast %vocab_i64 : i64 to index
    %logits_init = tensor.empty(%c1, %vocab) : tensor<?x?xf16>
    %logits_zero = linalg.fill ins(%cst : f16) outs(%logits_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %result:2 = scf.for %pi = %c0 to %arg1 step %c1
        iter_args(%pc = %cache, %prev_logits = %logits_zero) -> (!util.list<?>, tensor<?x?xf16>) {
      %tok_val = tensor.extract %arg0[%pi] : tensor<?xi64>
      %tok_t_e = tensor.empty() : tensor<1xi64>
      %tok_t = tensor.insert %tok_val into %tok_t_e[%c0] : tensor<1xi64>
      %tok = tensor.cast %tok_t : tensor<1xi64> to tensor<?xi64>
      %pos_val = arith.index_cast %pi : index to i64
      %pos_t_e = tensor.empty() : tensor<1xi64>
      %pos_t = tensor.insert %pos_val into %pos_t_e[%c0] : tensor<1xi64>
      %pos = tensor.cast %pos_t : tensor<1xi64> to tensor<?xi64>
      // ctx_len for this position = pi (number of previous tokens already in cache)
      // At position 0, ctx_len=0 (no prior context), at position 1, ctx_len=1, etc.
      %dec:2 = util.call @decode(%tok, %pos, %pc, %max_seq_len_val, %pi) : (tensor<?xi64>, tensor<?xi64>, !util.list<?>, index, index) -> (tensor<?x?xf16>, !util.list<?>)
      scf.yield %dec#1, %dec#0 : !util.list<?>, tensor<?x?xf16>
    }
    util.return %result#1, %result#0 : tensor<?x?xf16>, !util.list<?>
  }
  util.func private @attention_prefill(
      %arg0: tensor<?x?x?x?xf16>,  // Q [batch, n_heads, seq_len, head_dim]
      %arg1: tensor<?x?x?x?xf16>,  // K [batch, n_kv_heads, seq_len, head_dim]
      %arg2: tensor<?x?x?x?xf16>,  // V [batch, n_kv_heads, seq_len, head_dim]
      %arg3: f32                     // scale = 1/sqrt(head_dim)
  ) -> tensor<?x?x?x?xf16> {       // [batch, n_heads, seq_len, head_dim]
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index

    // Q dims: [batch, n_heads, seq_len, head_dim]
    %batch = tensor.dim %arg0, %c0 : tensor<?x?x?x?xf16>
    %n_heads = tensor.dim %arg0, %c1 : tensor<?x?x?x?xf16>
    %seq_len_q = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf16>
    %head_dim = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf16>

    // K dims
    %n_kv_heads = tensor.dim %arg1, %c1 : tensor<?x?x?x?xf16>
    %seq_len_k = tensor.dim %arg1, %c2 : tensor<?x?x?x?xf16>

    // Transpose Q from [batch, heads, seq, dim] to [batch, seq, heads, dim]
    %q_t_init = tensor.empty(%batch, %seq_len_q, %n_heads, %head_dim) : tensor<?x?x?x?xf16>
    %q_transposed = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%arg0 : tensor<?x?x?x?xf16>) outs(%q_t_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    // Transpose K from [batch, kv_heads, seq, dim] to [batch, seq, kv_heads, dim]
    %k_t_init = tensor.empty(%batch, %seq_len_k, %n_kv_heads, %head_dim) : tensor<?x?x?x?xf16>
    %k_transposed = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%arg1 : tensor<?x?x?x?xf16>) outs(%k_t_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    // Transpose V from [batch, kv_heads, seq, dim] to [batch, seq, kv_heads, dim]
    %v_t_init = tensor.empty(%batch, %seq_len_k, %n_kv_heads, %head_dim) : tensor<?x?x?x?xf16>
    %v_transposed = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%arg2 : tensor<?x?x?x?xf16>) outs(%v_t_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    // GQA: expand K from [batch, seq, n_kv_heads, head_dim]
    //   to [batch, seq, n_kv_heads, gqa_ratio, head_dim]
    //   then collapse to [batch, seq, n_heads, head_dim]
    %gqa_ratio = arith.divui %n_heads, %n_kv_heads : index

    %k_exp_init = tensor.empty(%batch, %seq_len_k, %n_kv_heads, %gqa_ratio, %head_dim) : tensor<?x?x?x?x?xf16>
    %k_expanded = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%k_transposed : tensor<?x?x?x?xf16>) outs(%k_exp_init : tensor<?x?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?x?xf16>
    %k_full = tensor.collapse_shape %k_expanded [[0], [1], [2, 3], [4]]
        : tensor<?x?x?x?x?xf16> into tensor<?x?x?x?xf16>

    %v_exp_init = tensor.empty(%batch, %seq_len_k, %n_kv_heads, %gqa_ratio, %head_dim) : tensor<?x?x?x?x?xf16>
    %v_expanded = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
    } ins(%v_transposed : tensor<?x?x?x?xf16>) outs(%v_exp_init : tensor<?x?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?x?xf16>
    %v_full = tensor.collapse_shape %v_expanded [[0], [1], [2, 3], [4]]
        : tensor<?x?x?x?x?xf16> into tensor<?x?x?x?xf16>

    // Now Q, K_full, V_full are all [batch, seq, n_heads, head_dim]
    // Collapse batch*n_heads for iree_linalg_ext.attention: [batch*n_heads, seq, head_dim]
    // Rearrange: [batch, seq, n_heads, dim] -> collapse batch,seq -> no
    // iree_linalg_ext.attention expects [batch, seq, dim] where batch includes head dim
    // So: [batch, seq, n_heads, dim] -> [batch*n_heads, seq, dim]
    // Need to permute first: [batch, n_heads, seq, dim] then collapse [batch*n_heads, seq, dim]
    // q_transposed is [batch, seq, n_heads, dim]
    // Permute to [batch, n_heads, seq, dim]:
    %q_perm_init = tensor.empty(%batch, %n_heads, %seq_len_q, %head_dim) : tensor<?x?x?x?xf16>
    %q_perm = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%q_transposed : tensor<?x?x?x?xf16>) outs(%q_perm_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    %k_perm_init = tensor.empty(%batch, %n_heads, %seq_len_k, %head_dim) : tensor<?x?x?x?xf16>
    %k_perm = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%k_full : tensor<?x?x?x?xf16>) outs(%k_perm_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    %v_perm_init = tensor.empty(%batch, %n_heads, %seq_len_k, %head_dim) : tensor<?x?x?x?xf16>
    %v_perm = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%v_full : tensor<?x?x?x?xf16>) outs(%v_perm_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    %bnh = arith.muli %batch, %n_heads : index
    %q_3d = tensor.collapse_shape %q_perm [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %k_3d = tensor.collapse_shape %k_perm [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    %v_3d = tensor.collapse_shape %v_perm [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>

    // Causal mask: [batch*n_heads, seq_len_q, seq_len_k]
    // mask[b, i, j] = (j <= i)  — lower triangular
    %mask_init = tensor.empty(%bnh, %seq_len_q, %seq_len_k) : tensor<?x?x?xi1>
    %causal_mask = linalg.generic {
        indexing_maps = [#map8],
        iterator_types = ["parallel", "parallel", "parallel"]
    } outs(%mask_init : tensor<?x?x?xi1>) {
    ^bb0(%out: i1):
      %qi = linalg.index 1 : index
      %ki = linalg.index 2 : index
      %cmp = arith.cmpi ule, %ki, %qi : index
      linalg.yield %cmp : i1
    } -> tensor<?x?x?xi1>

    // Call iree_linalg_ext.attention with causal mask
    %out_init = tensor.empty(%bnh, %seq_len_q, %head_dim) : tensor<?x?x?xf16>
    %attn_out = iree_linalg_ext.attention {
        indexing_maps = [#map9, #map10, #map11, #map12, #map13, #map14]
    } ins(%q_3d, %k_3d, %v_3d, %arg3, %causal_mask
        : tensor<?x?x?xf16>, tensor<?x?x?xf16>, tensor<?x?x?xf16>, f32, tensor<?x?x?xi1>)
      outs(%out_init : tensor<?x?x?xf16>) {
    ^bb0(%score: f32):
      iree_linalg_ext.yield %score : f32
    } -> tensor<?x?x?xf16>

    // Reshape: [batch*n_heads, seq, head_dim] -> [batch, n_heads, seq, head_dim]
    %out_4d = tensor.expand_shape %attn_out [[0, 1], [2], [3]]
        output_shape [%batch, %n_heads, %seq_len_q, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>

    util.return %out_4d : tensor<?x?x?x?xf16>
  }
  // ---- Transformer layer prefill with direct cache ----
  util.func private @transformer_layer_prefill(
      %hidden: tensor<?x?xf16>,           // [seq_len, hidden_dim]
      %positions: tensor<?xi64>,           // [seq_len]
      %cache: !util.list<?>,               // KV cache
      %max_seq_len_val: index,             // max_seq_len for slot calculation
      %start_pos: index,                   // starting position in cache
      %layer_idx: i32,                      // layer index
      %n_heads_idx: index,                 // 16
      %n_kv_heads_idx: index,              // 8
      %hidden_dim: index,                  // 1024
      %ffn_dim: index,                     // 3072
      %rms_eps: f32,                       // 1e-6
      %rope_base: f32,                     // 1e6
      %rope_scale: f32                     // 1.0
  ) -> (tensor<?x?xf16>, !util.list<?>) { // [seq_len, hidden_dim], updated cache
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst_zero_f16 = arith.constant 0.000000e+00 : f16
    %cst_zero_f32 = arith.constant 0.000000e+00 : f32

    %seq_len = tensor.dim %hidden, %c0 : tensor<?x?xf16>
    %layer_idx_index = arith.index_cast %layer_idx : i32 to index

    // head_dim from hparams (128 for Qwen3 — NOT hidden_dim/n_heads which gives 64)
    %head_dim_i64 = util.call @hparams.head_dim() : () -> i64
    %head_dim = arith.index_cast %head_dim_i64 : i64 to index
    // kv_dim = n_kv_heads * head_dim = 8 * 128 = 1024
    %kv_dim = arith.muli %n_kv_heads_idx, %head_dim : index
    // q_proj_dim = n_heads * head_dim = 16 * 128 = 2048
    %q_proj_dim = arith.muli %n_heads_idx, %head_dim : index

    // --- Load weights ---
    %attn_norm_w = util.call @model_params.attn_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %ffn_norm_w = util.call @model_params.ffn_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %q_weight = util.call @model_params.attn_q_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %k_weight = util.call @model_params.attn_k_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %v_weight = util.call @model_params.attn_v_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %o_weight = util.call @model_params.attn_output_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %q_norm_w = util.call @model_params.attn_q_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %k_norm_w = util.call @model_params.attn_k_norm_weight(%layer_idx) : (i32) -> tensor<?xf16>
    %gate_up_weight = util.call @model_params.ffn_gate_up_weight(%layer_idx) : (i32) -> tensor<?x?xf16>
    %down_weight = util.call @model_params.ffn_down_weight(%layer_idx) : (i32) -> tensor<?x?xf16>

    // ---- Attention sub-block ----

    // 1. RMS norm on input
    %normed = util.call @rms_norm_components.rms_norm_linalg(%hidden, %attn_norm_w, %rms_eps) : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // 2. QKV projection: [seq_len, hidden_dim] @ [hidden_dim, proj_dim]
    %q_init = tensor.empty(%seq_len, %q_proj_dim) : tensor<?x?xf16>
    %q_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%q_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %q_proj = linalg.matmul ins(%normed, %q_weight : tensor<?x?xf16>, tensor<?x?xf16>) outs(%q_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    %k_init = tensor.empty(%seq_len, %kv_dim) : tensor<?x?xf16>
    %k_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%k_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %k_proj = linalg.matmul ins(%normed, %k_weight : tensor<?x?xf16>, tensor<?x?xf16>) outs(%k_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    %v_init = tensor.empty(%seq_len, %kv_dim) : tensor<?x?xf16>
    %v_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%v_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %v_proj = linalg.matmul ins(%normed, %v_weight : tensor<?x?xf16>, tensor<?x?xf16>) outs(%v_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // 3. Reshape to [seq_len, n_heads/n_kv_heads, head_dim]
    %q_3d = tensor.expand_shape %q_proj [[0], [1, 2]]
        output_shape [%seq_len, %n_heads_idx, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>
    %k_3d = tensor.expand_shape %k_proj [[0], [1, 2]]
        output_shape [%seq_len, %n_kv_heads_idx, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>
    %v_3d = tensor.expand_shape %v_proj [[0], [1, 2]]
        output_shape [%seq_len, %n_kv_heads_idx, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>

    // 4. QK norm (RMS norm per-head on Q and K) BEFORE RoPE
    //    Flatten [seq_len, n_heads, head_dim] -> [seq_len*n_heads, head_dim]
    //    Apply RMS norm, then reshape back
    %q_bh = arith.muli %seq_len, %n_heads_idx : index
    %q_flat = tensor.collapse_shape %q_3d [[0, 1], [2]]
        : tensor<?x?x?xf16> into tensor<?x?xf16>

    // Q norm via flow.dispatch.region (fuses reduction + elementwise into 1 dispatch)
    %q_normed_flat = flow.dispatch.region[] -> (tensor<?x?xf16>{%q_bh, %head_dim}) {
      %_si_q = tensor.empty(%q_bh) : tensor<?xf32>
      %_sz_q = linalg.fill ins(%cst_zero_f32 : f32) outs(%_si_q : tensor<?xf32>) -> tensor<?xf32>
      %_hdi_q = arith.index_cast %head_dim : index to i32
      %_hdf_q = arith.sitofp %_hdi_q : i32 to f32
      %_ss_q = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]}
          ins(%q_flat : tensor<?x?xf16>) outs(%_sz_q : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %a = arith.extf %in : f16 to f32
        %b = arith.mulf %a, %a : f32
        %c = arith.addf %out, %b : f32
        linalg.yield %c : f32
      } -> tensor<?xf32>
      %_oi_q = tensor.empty(%q_bh, %head_dim) : tensor<?x?xf16>
      %_on_q = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]}
          ins(%q_flat, %_ss_q, %q_norm_w : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>) outs(%_oi_q : tensor<?x?xf16>) {
      ^bb0(%in: f16, %ss: f32, %w: f16, %out: f16):
        %a = arith.divf %ss, %_hdf_q : f32
        %b = arith.addf %a, %rms_eps : f32
        %c = math.sqrt %b : f32
        %d = arith.extf %in : f16 to f32
        %e = arith.extf %w : f16 to f32
        %f = arith.divf %d, %c : f32
        %g = arith.mulf %f, %e : f32
        %h = arith.truncf %g : f32 to f16
        linalg.yield %h : f16
      } -> tensor<?x?xf16>
      flow.return %_on_q : tensor<?x?xf16>
    }

    %k_bh = arith.muli %seq_len, %n_kv_heads_idx : index
    %k_flat = tensor.collapse_shape %k_3d [[0, 1], [2]]
        : tensor<?x?x?xf16> into tensor<?x?xf16>

    // K norm via flow.dispatch.region
    %k_normed_flat = flow.dispatch.region[] -> (tensor<?x?xf16>{%k_bh, %head_dim}) {
      %_si_k = tensor.empty(%k_bh) : tensor<?xf32>
      %_sz_k = linalg.fill ins(%cst_zero_f32 : f32) outs(%_si_k : tensor<?xf32>) -> tensor<?xf32>
      %_hdi_k = arith.index_cast %head_dim : index to i32
      %_hdf_k = arith.sitofp %_hdi_k : i32 to f32
      %_ss_k = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]}
          ins(%k_flat : tensor<?x?xf16>) outs(%_sz_k : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %a = arith.extf %in : f16 to f32
        %b = arith.mulf %a, %a : f32
        %c = arith.addf %out, %b : f32
        linalg.yield %c : f32
      } -> tensor<?xf32>
      %_oi_k = tensor.empty(%k_bh, %head_dim) : tensor<?x?xf16>
      %_on_k = linalg.generic {indexing_maps = [#map, #map1, #map3, #map], iterator_types = ["parallel", "parallel"]}
          ins(%k_flat, %_ss_k, %k_norm_w : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf16>) outs(%_oi_k : tensor<?x?xf16>) {
      ^bb0(%in: f16, %ss: f32, %w: f16, %out: f16):
        %a = arith.divf %ss, %_hdf_k : f32
        %b = arith.addf %a, %rms_eps : f32
        %c = math.sqrt %b : f32
        %d = arith.extf %in : f16 to f32
        %e = arith.extf %w : f16 to f32
        %f = arith.divf %d, %c : f32
        %g = arith.mulf %f, %e : f32
        %h = arith.truncf %g : f32 to f16
        linalg.yield %h : f16
      } -> tensor<?x?xf16>
      flow.return %_on_k : tensor<?x?xf16>
    }

    // Reshape back to [seq_len, n_heads/n_kv_heads, head_dim]
    %q_normed_3d = tensor.expand_shape %q_normed_flat [[0, 1], [2]]
        output_shape [%seq_len, %n_heads_idx, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>
    %k_normed_3d = tensor.expand_shape %k_normed_flat [[0, 1], [2]]
        output_shape [%seq_len, %n_kv_heads_idx, %head_dim]
        : tensor<?x?xf16> into tensor<?x?x?xf16>

    // 5. Reshape for RoPE: [batch=1, seq_len, n_heads, head_dim]
    %c1_idx = arith.constant 1 : index
    %q_4d = tensor.expand_shape %q_normed_3d [[0, 1], [2], [3]]
        output_shape [%c1_idx, %seq_len, %n_heads_idx, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %k_4d = tensor.expand_shape %k_normed_3d [[0, 1], [2], [3]]
        output_shape [%c1_idx, %seq_len, %n_kv_heads_idx, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>

    // Build positions tensor for RoPE: [1, seq_len] from the input positions [seq_len]
    %pos_2d = tensor.expand_shape %positions [[0, 1]]
        output_shape [%c1_idx, %seq_len]
        : tensor<?xi64> into tensor<?x?xi64>

    // 6. Apply RoPE
    %q_roped = util.call @position_components.rope(%q_4d, %pos_2d, %rope_base, %rope_scale) : (tensor<?x?x?x?xf16>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf16>
    %k_roped = util.call @position_components.rope(%k_4d, %pos_2d, %rope_base, %rope_scale) : (tensor<?x?x?x?xf16>, tensor<?x?xi64>, f32, f32) -> tensor<?x?x?x?xf16>

    // 7. Transpose for attention: [1, seq_len, heads, dim] -> [1, heads, seq_len, dim]
    %q_attn_init = tensor.empty(%c1_idx, %n_heads_idx, %seq_len, %head_dim) : tensor<?x?x?x?xf16>
    %q_attn = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%q_roped : tensor<?x?x?x?xf16>) outs(%q_attn_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    %k_attn_init = tensor.empty(%c1_idx, %n_kv_heads_idx, %seq_len, %head_dim) : tensor<?x?x?x?xf16>
    %k_attn = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%k_roped : tensor<?x?x?x?xf16>) outs(%k_attn_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    // V: reshape [seq_len, n_kv_heads, head_dim] -> [1, seq_len, n_kv_heads, head_dim]
    //    then transpose to [1, n_kv_heads, seq_len, head_dim]
    %v_4d = tensor.expand_shape %v_3d [[0, 1], [2], [3]]
        output_shape [%c1_idx, %seq_len, %n_kv_heads_idx, %head_dim]
        : tensor<?x?x?xf16> into tensor<?x?x?x?xf16>
    %v_attn_init = tensor.empty(%c1_idx, %n_kv_heads_idx, %seq_len, %head_dim) : tensor<?x?x?x?xf16>
    %v_attn = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%v_4d : tensor<?x?x?x?xf16>) outs(%v_attn_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>

    // 8. Compute scale = 1/sqrt(head_dim)
    %head_dim_i32 = arith.index_cast %head_dim : index to i32
    %head_dim_f32 = arith.sitofp %head_dim_i32 : i32 to f32
    %scale = math.rsqrt %head_dim_f32 : f32

    // 9. Call attention_prefill
    %attn_out = util.call @attention_prefill(%q_attn, %k_attn, %v_attn, %scale)
        : (tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, tensor<?x?x?x?xf16>, f32) -> tensor<?x?x?x?xf16>

    // 10. Reshape attention output: [1, n_heads, seq_len, head_dim]
    //     -> transpose to [1, seq_len, n_heads, head_dim]
    //     -> collapse to [seq_len, n_heads*head_dim]
    %attn_t_init = tensor.empty(%c1_idx, %seq_len, %n_heads_idx, %head_dim) : tensor<?x?x?x?xf16>
    %attn_transposed = linalg.generic {
        indexing_maps = [#map4, #map5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%attn_out : tensor<?x?x?x?xf16>) outs(%attn_t_init : tensor<?x?x?x?xf16>) {
    ^bb0(%in: f16, %out: f16):
      linalg.yield %in : f16
    } -> tensor<?x?x?x?xf16>
    // [1, seq_len, n_heads, head_dim] -> [seq_len, n_heads*head_dim]
    %attn_flat = tensor.collapse_shape %attn_transposed [[0, 1], [2, 3]]
        : tensor<?x?x?x?xf16> into tensor<?x?xf16>

    // 11. Output projection: [seq_len, n_heads*head_dim] @ [n_heads*head_dim, hidden_dim]
    %o_init = tensor.empty(%seq_len, %hidden_dim) : tensor<?x?xf16>
    %o_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%o_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %attn_output = linalg.matmul ins(%attn_flat, %o_weight : tensor<?x?xf16>, tensor<?x?xf16>) outs(%o_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // 12. Scatter K/V into cache for ALL positions using direct scatter
    // K after RoPE: [1, seq_len, n_kv_heads, head_dim] -> [seq_len, n_kv_heads, head_dim]
    %k_for_cache = tensor.collapse_shape %k_roped [[0, 1], [2], [3]]
        : tensor<?x?x?x?xf16> into tensor<?x?x?xf16>
    // v_3d is already [seq_len, n_kv_heads, head_dim]
    %cache_updated = util.call @kvcache_direct_scatter_prefill(
        %cache, %layer_idx_index, %k_for_cache, %v_3d, %max_seq_len_val, %start_pos)
        : (!util.list<?>, index, tensor<?x?x?xf16>, tensor<?x?x?xf16>, index, index) -> !util.list<?>

    // 13. Residual connection: hidden + attn_output
    %res1_init = tensor.empty(%seq_len, %hidden_dim) : tensor<?x?xf16>
    %residual1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%hidden, %attn_output : tensor<?x?xf16>, tensor<?x?xf16>) outs(%res1_init : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %r = arith.addf %in, %in_1 : f16
      linalg.yield %r : f16
    } -> tensor<?x?xf16>

    // ---- FFN sub-block ----

    // 14. RMS norm
    %ffn_normed = util.call @rms_norm_components.rms_norm_linalg(%residual1, %ffn_norm_w, %rms_eps) : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // 15. Gate+Up projection: [seq_len, hidden_dim] @ [hidden_dim, 2*ffn_dim]
    %ffn_2x = arith.muli %ffn_dim, %c2 : index
    %gu_init = tensor.empty(%seq_len, %ffn_2x) : tensor<?x?xf16>
    %gu_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%gu_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %gate_up = linalg.matmul ins(%ffn_normed, %gate_up_weight : tensor<?x?xf16>, tensor<?x?xf16>) outs(%gu_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // 16. SwiGLU: split gate and up, apply silu(gate) * up
    %gate_slice = tensor.extract_slice %gate_up[0, 0] [%seq_len, %ffn_dim] [1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>
    %up_slice = tensor.extract_slice %gate_up[0, %ffn_dim] [%seq_len, %ffn_dim] [1, 1]
        : tensor<?x?xf16> to tensor<?x?xf16>

    %swiglu_init = tensor.empty(%seq_len, %ffn_dim) : tensor<?x?xf16>
    %swiglu = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%gate_slice, %up_slice : tensor<?x?xf16>, tensor<?x?xf16>) outs(%swiglu_init : tensor<?x?xf16>) {
    ^bb0(%gate: f16, %up: f16, %out: f16):
      // SiLU(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
      %neg_gate = arith.negf %gate : f16
      %exp_neg = math.exp %neg_gate : f16
      %one = arith.constant 1.000000e+00 : f16
      %denom = arith.addf %one, %exp_neg : f16
      %sigmoid = arith.divf %one, %denom : f16
      %silu = arith.mulf %gate, %sigmoid : f16
      %result = arith.mulf %silu, %up : f16
      linalg.yield %result : f16
    } -> tensor<?x?xf16>

    // 17. Down projection: [seq_len, ffn_dim] @ [ffn_dim, hidden_dim]
    %down_init = tensor.empty(%seq_len, %hidden_dim) : tensor<?x?xf16>
    %down_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%down_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %ffn_out = linalg.matmul ins(%swiglu, %down_weight : tensor<?x?xf16>, tensor<?x?xf16>) outs(%down_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    // 18. Residual connection: residual1 + ffn_out
    %res2_init = tensor.empty(%seq_len, %hidden_dim) : tensor<?x?xf16>
    %residual2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%residual1, %ffn_out : tensor<?x?xf16>, tensor<?x?xf16>) outs(%res2_init : tensor<?x?xf16>) {
    ^bb0(%in: f16, %in_1: f16, %out: f16):
      %r = arith.addf %in, %in_1 : f16
      linalg.yield %r : f16
    } -> tensor<?x?xf16>

    util.return %residual2, %cache_updated : tensor<?x?xf16>, !util.list<?>
  }
  // ---- prefill: direct cache ----
  util.func public @prefill(
      %tokens: tensor<?xi64>,             // [seq_len] input token IDs
      %seq_len_val: index,                 // seq_len (dynamic)
      %cache: !util.list<?>,               // KV cache (pre-allocated)
      %max_seq_len_val: index,             // max_seq_len for slot calculation
      %start_pos: index                    // starting position in cache (0 for first turn)
  ) -> (tensor<?x?xf16>, !util.list<?>) { // [1, vocab_size] logits, updated cache
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Load hyperparams
    %vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %n_layers_i64 = util.call @hparams.block_count() : () -> i64
    %hidden_dim_i64 = util.call @hparams.embedding_length() : () -> i64
    %n_heads_i64 = util.call @hparams.attention_head_count() : () -> i64
    %n_kv_heads_i64 = util.call @hparams.attention_head_count_kv() : () -> i64
    %ffn_dim_i64 = util.call @hparams.feed_forward_length() : () -> i64
    %rope_base = util.call @hparams.rope_freq_base() : () -> f32
    %rms_eps = util.call @hparams.layer_norm_rms_epsilon() : () -> f32

    %vocab = arith.index_cast %vocab_i64 : i64 to index
    %n_layers = arith.index_cast %n_layers_i64 : i64 to index
    %hidden_dim = arith.index_cast %hidden_dim_i64 : i64 to index
    %n_heads = arith.index_cast %n_heads_i64 : i64 to index
    %n_kv_heads = arith.index_cast %n_kv_heads_i64 : i64 to index
    %ffn_dim = arith.index_cast %ffn_dim_i64 : i64 to index

    %rope_scale = arith.constant 1.000000e+00 : f32
    %cst_zero_f16 = arith.constant 0.000000e+00 : f16

    // 1. Embedding lookup for ALL tokens: [seq_len] -> [seq_len, hidden_dim]
    %embd_weight = util.call @model_params.token_embd_weight() : () -> tensor<?x?xf16>
    %hidden_init = util.call @embedding_components.embedding_lookup_1d(%embd_weight, %tokens)
        : (tensor<?x?xf16>, tensor<?xi64>) -> tensor<?x?xf16>

    // 2. Build positions tensor: [0, 1, 2, ..., seq_len-1] offset by start_pos
    %pos_init = tensor.empty(%seq_len_val) : tensor<?xi64>
    %positions = linalg.generic {
        indexing_maps = [#map2],
        iterator_types = ["parallel"]
    } outs(%pos_init : tensor<?xi64>) {
    ^bb0(%out: i64):
      %idx = linalg.index 0 : index
      %abs_pos = arith.addi %idx, %start_pos : index
      %val = arith.index_cast %abs_pos : index to i64
      linalg.yield %val : i64
    } -> tensor<?xi64>

    // 3. Loop over 28 layers
    %result:2 = scf.for %layer_iv = %c0 to %n_layers step %c1
        iter_args(%h = %hidden_init, %c = %cache) -> (tensor<?x?xf16>, !util.list<?>) {
      %li32 = arith.index_cast %layer_iv : index to i32
      %layer_out:2 = util.call @transformer_layer_prefill(
          %h, %positions, %c, %max_seq_len_val, %start_pos, %li32,
          %n_heads, %n_kv_heads, %hidden_dim, %ffn_dim,
          %rms_eps, %rope_base, %rope_scale)
          : (tensor<?x?xf16>, tensor<?xi64>, !util.list<?>, index, index, i32,
             index, index, index, index, f32, f32, f32) -> (tensor<?x?xf16>, !util.list<?>)
      scf.yield %layer_out#0, %layer_out#1 : tensor<?x?xf16>, !util.list<?>
    }

    // 4. Final RMS norm on full hidden [seq_len, hidden_dim]
    %output_norm_w = util.call @model_params.output_norm_weight() : () -> tensor<?xf16>
    %normed_final = util.call @rms_norm_components.rms_norm_linalg(%result#0, %output_norm_w, %rms_eps)
        : (tensor<?x?xf16>, tensor<?xf16>, f32) -> tensor<?x?xf16>

    // 5. Extract LAST token's hidden state: [seq_len, hidden_dim] -> [1, hidden_dim]
    %last_idx = arith.subi %seq_len_val, %c1 : index
    %last_hidden = tensor.extract_slice %normed_final[%last_idx, 0] [1, %hidden_dim] [1, 1]
        : tensor<?x?xf16> to tensor<1x?xf16>
    %last_hidden_dyn = tensor.cast %last_hidden : tensor<1x?xf16> to tensor<?x?xf16>

    // 6. Output projection: [1, hidden_dim] @ [hidden_dim, vocab_size] -> [1, vocab_size]
    %output_w = util.call @model_params.output_weight() : () -> tensor<?x?xf16>
    %logits_init = tensor.empty(%c1, %vocab) : tensor<?x?xf16>
    %logits_zero = linalg.fill ins(%cst_zero_f16 : f16) outs(%logits_init : tensor<?x?xf16>) -> tensor<?x?xf16>
    %logits = linalg.matmul ins(%last_hidden_dyn, %output_w : tensor<?x?xf16>, tensor<?x?xf16>) outs(%logits_zero : tensor<?x?xf16>) -> tensor<?x?xf16>

    util.return %logits, %result#1 : tensor<?x?xf16>, !util.list<?>
  }
  // ---- run: direct cache ----
  util.func public @run(%arg0: tensor<?xi64>, %arg1: index, %arg2: index, %arg3: i64) -> (tensor<?xi64>, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %total = arith.addi %arg1, %arg2 : index
    %cache_init = util.call @allocate_kv_cache(%total) : (index) -> !util.list<?>
    // Initialize global KV cache tensors (zero-filled)
    %cst_zero = arith.constant 0.000000e+00 : f16
    %g_k_init = tensor.empty() : tensor<14336x8x128xf16>
    %g_k_fill = linalg.fill ins(%cst_zero : f16) outs(%g_k_init : tensor<14336x8x128xf16>) -> tensor<14336x8x128xf16>
    util.global.store %g_k_fill, @g_k_cache : tensor<14336x8x128xf16>
    %g_v_init = tensor.empty() : tensor<14336x8x128xf16>
    %g_v_fill = linalg.fill ins(%cst_zero : f16) outs(%g_v_init : tensor<14336x8x128xf16>) -> tensor<14336x8x128xf16>
    util.global.store %g_v_fill, @g_v_cache : tensor<14336x8x128xf16>
    // Parallel prefill — processes ALL tokens at once (not one-by-one)
    %prefill:2 = util.call @prefill(%arg0, %arg1, %cache_init, %total, %c0) : (tensor<?xi64>, index, !util.list<?>, index, index) -> (tensor<?x?xf16>, !util.list<?>)
    // Argmax on last prefill logits
    %cst_neg_inf = arith.constant 0xFC00 : f16
    %c_neg1_i64 = arith.constant -1 : i64
    %am_val_init = tensor.empty() : tensor<f16>
    %am_idx_init = tensor.empty() : tensor<i64>
    %am_val = linalg.fill ins(%cst_neg_inf : f16) outs(%am_val_init : tensor<f16>) -> tensor<f16>
    %am_idx = linalg.fill ins(%c_neg1_i64 : i64) outs(%am_idx_init : tensor<i64>) -> tensor<i64>
    %vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %vocab = arith.index_cast %vocab_i64 : i64 to index
    %logits_row = tensor.extract_slice %prefill#0[0, 0] [1, %vocab] [1, 1] : tensor<?x?xf16> to tensor<?xf16>
    %argmax:2 = linalg.generic {indexing_maps = [#map2, #map16, #map16], iterator_types = ["reduction"]} ins(%logits_row : tensor<?xf16>) outs(%am_val, %am_idx : tensor<f16>, tensor<i64>) {
    ^bb0(%in: f16, %out: f16, %out_idx: i64):
      %idx_val = linalg.index 0 : index
      %idx_i64 = arith.index_cast %idx_val : index to i64
      %is_gt = arith.cmpf ogt, %in, %out : f16
      %new_val = arith.select %is_gt, %in, %out : f16
      %new_idx = arith.select %is_gt, %idx_i64, %out_idx : i64
      linalg.yield %new_val, %new_idx : f16, i64
    } -> (tensor<f16>, tensor<i64>)
    %first_tok = tensor.extract %argmax#1[] : tensor<i64>
    %last_pos = arith.index_cast %arg1 : index to i64
    // Copy prefill cache (list) into global tensors for the generate loop
    %pre_k_bv = util.list.get %prefill#1[%c0] : !util.list<?> -> !hal.buffer_view
    %pre_v_bv = util.list.get %prefill#1[%c1] : !util.list<?> -> !hal.buffer_view
    %pre_total = hal.buffer_view.dim<%pre_k_bv : !hal.buffer_view>[0] : index
    %pre_k = hal.tensor.import %pre_k_bv : !hal.buffer_view -> tensor<?x8x128xf16>{%pre_total}
    %pre_v = hal.tensor.import %pre_v_bv : !hal.buffer_view -> tensor<?x8x128xf16>{%pre_total}
    // Pad to global size (14336) and store
    %cst_pad = arith.constant 0.000000e+00 : f16
    %g_k_pad = tensor.empty() : tensor<14336x8x128xf16>
    %g_k_z = linalg.fill ins(%cst_pad : f16) outs(%g_k_pad : tensor<14336x8x128xf16>) -> tensor<14336x8x128xf16>
    %g_k_set = tensor.insert_slice %pre_k into %g_k_z[0, 0, 0] [%pre_total, 8, 128] [1, 1, 1] : tensor<?x8x128xf16> into tensor<14336x8x128xf16>
    util.global.store %g_k_set, @g_k_cache : tensor<14336x8x128xf16>
    %g_v_pad = tensor.empty() : tensor<14336x8x128xf16>
    %g_v_z = linalg.fill ins(%cst_pad : f16) outs(%g_v_pad : tensor<14336x8x128xf16>) -> tensor<14336x8x128xf16>
    %g_v_set = tensor.insert_slice %pre_v into %g_v_z[0, 0, 0] [%pre_total, 8, 128] [1, 1, 1] : tensor<?x8x128xf16> into tensor<14336x8x128xf16>
    util.global.store %g_v_set, @g_v_cache : tensor<14336x8x128xf16>
    %gen:3 = util.call @generate(%first_tok, %prefill#1, %total, %arg2, %arg3, %last_pos) : (i64, !util.list<?>, index, index, i64, i64) -> (tensor<?xi64>, index, !util.list<?>)
    util.return %gen#0, %gen#1 : tensor<?xi64>, index
  }

  // ---- Multi-turn chat with persistent KV cache ----

  // Initialize cache for a conversation (call once)
  util.func public @init_chat(%max_ctx: index) {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    %cache = util.call @allocate_kv_cache(%max_ctx) : (index) -> !util.list<?>
    util.global.store %cache, @kv_cache : !util.list<?>
    util.global.store %max_ctx, @max_seq_len : index
    util.global.store %c0, @current_pos : index
    util.global.store %true, @is_initialized : i1
    util.return
  }

  // One chat turn: prefill new tokens + generate. All state in globals.
  util.func public @chat_turn(%new_tokens: tensor<?xi64>, %n_tokens: index, %max_gen: index, %eos: i64) -> (tensor<?xi64>, index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_zero = arith.constant 0.000000e+00 : f16
    %cst_neg_inf = arith.constant 0xFC00 : f16
    %c_neg1_i64 = arith.constant -1 : i64
    %cache = util.global.load @kv_cache : !util.list<?>
    %mslen = util.global.load @max_seq_len : index
    %pos = util.global.load @current_pos : index
    %vocab_i64 = util.call @hparams.vocab_size() : () -> i64
    %vocab = arith.index_cast %vocab_i64 : i64 to index
    // Parallel prefill: process ALL new tokens at once
    %pfill:2 = util.call @prefill(%new_tokens, %n_tokens, %cache, %mslen, %pos) : (tensor<?xi64>, index, !util.list<?>, index, index) -> (tensor<?x?xf16>, !util.list<?>)
    %new_pos = arith.addi %pos, %n_tokens : index
    %logits_row = tensor.extract_slice %pfill#0[0, 0] [1, %vocab] [1, 1] : tensor<?x?xf16> to tensor<?xf16>
    %am_v_i = tensor.empty() : tensor<f16>
    %am_x_i = tensor.empty() : tensor<i64>
    %am_v = linalg.fill ins(%cst_neg_inf : f16) outs(%am_v_i : tensor<f16>) -> tensor<f16>
    %am_x = linalg.fill ins(%c_neg1_i64 : i64) outs(%am_x_i : tensor<i64>) -> tensor<i64>
    %argmax:2 = linalg.generic {indexing_maps = [#map2, #map16, #map16], iterator_types = ["reduction"]} ins(%logits_row : tensor<?xf16>) outs(%am_v, %am_x : tensor<f16>, tensor<i64>) {
    ^bb0(%in: f16, %out: f16, %out_idx: i64):
      %idx = linalg.index 0 : index
      %idx_i64 = arith.index_cast %idx : index to i64
      %gt = arith.cmpf ogt, %in, %out : f16
      %nv = arith.select %gt, %in, %out : f16
      %ni = arith.select %gt, %idx_i64, %out_idx : i64
      linalg.yield %nv, %ni : f16, i64
    } -> (tensor<f16>, tensor<i64>)
    %first_tok = tensor.extract %argmax#1[] : tensor<i64>
    // Generate
    %gen_start = arith.index_cast %new_pos : index to i64
    %gen:3 = util.call @generate(%first_tok, %pfill#1, %mslen, %max_gen, %eos, %gen_start) : (i64, !util.list<?>, index, index, i64, i64) -> (tensor<?xi64>, index, !util.list<?>)
    // Update persistent state
    %gen_plus1 = arith.addi %gen#1, %c1 : index
    %final_pos = arith.addi %new_pos, %gen_plus1 : index
    util.global.store %gen#2, @kv_cache : !util.list<?>
    util.global.store %final_pos, @current_pos : index
    util.return %gen#0, %gen#1 : tensor<?xi64>, index
  }
}
