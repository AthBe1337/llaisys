#include "../../../device/nvidia/nvidia_common.cuh"


__global__ void linear_w8a16_kernel(
    cuda_bfloat16 *__restrict__ C,       // [M, N]
    const cuda_bfloat16 *__restrict__ A, // [M, K]
    const int8_t *__restrict__ B,      // [N, K] - INT8
    const cuda_bfloat16 *__restrict__ bias,
    const cuda_bfloat16 *__restrict__ scale,   // [N] - Per-channel scale
    const size_t M, const size_t N, const size_t K) {

    // Block tile sizes
    const int BM = 128;
    const int BN = 256;
    const int BK = 32;
    const int APAD = 8;
    const int BPAD = 8;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane_id = tid & 31;

    extern __shared__ cuda_bfloat16 smem_bf16[];
    cuda_bfloat16 *s_a = smem_bf16;
    cuda_bfloat16 *s_b = s_a + 2 * BM * (BK + APAD);
    
    size_t s_a_db_offset = BM * (BK + APAD);
    size_t s_b_db_offset = BN * (BK + APAD);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, cuda_bfloat16, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, cuda_bfloat16, wmma::col_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
    }

    // Indices calculation
    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid & 3) << 3;
    int load_b_smem_n = (tid >> 2) << 2;
    int load_b_smem_k = (tid & 3) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int comp_c_frag_m = wid & 1;
    int comp_c_frag_n = wid >> 1;

    // Prologue: Load first tile
    {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int gmem_m = load_a_gmem_m + i;
            int gmem_k = load_a_smem_k;
            int smem_m = load_a_smem_m + i;
            
            const cuda_bfloat16 *src_ptr = &A[OFFSET(gmem_m, gmem_k, K)];
            uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD)]);
            
            bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

            if (is_aligned) {
                 int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
                 int src_size = max(0, min(16, valid_bytes));
                 src_size = gmem_m < M ? src_size : 0;
                 asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(load_a_smem_addr), "l"(src_ptr), "r"(src_size));
            } else {
                // Fallback for unaligned
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_k + j < K && gmem_m < M) s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD)] = src_ptr[j];
                    else s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD)] = __float2bfloat16(0.0f);
                }
            }
        }

        // ==================== Load B (INT8 -> BF16) ====================
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int gmem_n = load_b_gmem_n + i;
            int gmem_k = load_b_smem_k;
            int smem_n = load_b_smem_n + i;

            float row_scale = 0.0f;
            if (gmem_n < N) {
                row_scale = __bfloat162float(scale[gmem_n]); 
            }
            
            const int8_t *src_ptr_int8 = &B[OFFSET(gmem_n, gmem_k, K)];
            
            int8_t loaded_w[8];
            
            if (gmem_n < N) {
                #pragma unroll
                for(int j=0; j<8; j++) {
                    if (gmem_k + j < K) loaded_w[j] = src_ptr_int8[j];
                    else loaded_w[j] = 0;
                }
            } else {
                #pragma unroll
                for(int j=0; j<8; j++) loaded_w[j] = 0;
            }

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float val_f = static_cast<float>(loaded_w[j]) * row_scale;
                s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD)] = __float2bfloat16(val_f);
            }
        }

        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    int num_k_tiles = div_ceil(K, BK);

    // Main Loop
    for (int bk = 1; bk < num_k_tiles; bk++) {
        int k_start = bk * BK;
        int curr_idx = (bk - 1) & 1;
        int next_idx = bk & 1;

        // ==================== Load Next A Tile (BF16) ====================
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int gmem_m = load_a_gmem_m + i;
            int gmem_k = k_start + load_a_smem_k;
            int smem_m = load_a_smem_m + i;

            const cuda_bfloat16 *src_ptr = &A[OFFSET(gmem_m, gmem_k, K)];
            // Offset logic for double buffer
            uint32_t load_a_smem_addr = __cvta_generic_to_shared(&s_a[OFFSET(smem_m, load_a_smem_k, BK + APAD) + next_idx * s_a_db_offset]);
            bool is_aligned = (reinterpret_cast<uint64_t>(src_ptr) % 16 == 0);

            if (is_aligned) {
                int valid_bytes = (K - gmem_k) * sizeof(cuda_bfloat16);
                int src_size = max(0, min(16, valid_bytes));
                src_size = gmem_m < M ? src_size : 0;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(load_a_smem_addr), "l"(src_ptr), "r"(src_size));
            } else {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    if (gmem_k + j < K && gmem_m < M) 
                        s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD) + next_idx * s_a_db_offset] = src_ptr[j];
                    else 
                        s_a[OFFSET(smem_m, load_a_smem_k + j, BK + APAD) + next_idx * s_a_db_offset] = __float2bfloat16(0.0f);
                }
            }
        }

        // ==================== Load Next B Tile (INT8 -> BF16) ====================
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int gmem_n = load_b_gmem_n + i;
            int gmem_k = k_start + load_b_smem_k;
            int smem_n = load_b_smem_n + i;
            
            float row_scale = 0.0f;
            if (gmem_n < N) {
                row_scale = __bfloat162float(scale[gmem_n]);
            }
            const int8_t *src_ptr_int8 = &B[OFFSET(gmem_n, gmem_k, K)];
            int8_t loaded_w[8];
            
            if (gmem_n < N) {
                #pragma unroll
                for(int j=0; j<8; j++) {
                    if (gmem_k + j < K) loaded_w[j] = src_ptr_int8[j];
                    else loaded_w[j] = 0;
                }
            } else {
                #pragma unroll
                for(int j=0; j<8; j++) loaded_w[j] = 0;
            }

            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float val_f = static_cast<float>(loaded_w[j]) * row_scale;
                s_b[OFFSET(smem_n, load_b_smem_k + j, BK + BPAD) + next_idx * s_b_db_offset] = __float2bfloat16(val_f);
            }
        }

        // ==================== Compute Current Tile ====================
        // Load fragments from Shared Memory (BF16)
        
        // Load A
        wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(comp_c_frag_m * 64, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(comp_c_frag_m * 64, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

        // Load B
        wmma::load_matrix_sync(frag_b[0][0], &s_b[OFFSET(comp_c_frag_n * 64, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[OFFSET(comp_c_frag_n * 64 + 16, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[OFFSET(comp_c_frag_n * 64 + 32, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[OFFSET(comp_c_frag_n * 64 + 48, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[OFFSET(comp_c_frag_n * 64, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[OFFSET(comp_c_frag_n * 64 + 16, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[OFFSET(comp_c_frag_n * 64 + 32, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[OFFSET(comp_c_frag_n * 64 + 48, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        // Commit async copy for the next tile
        asm("cp.async.commit_group;\n" ::);
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }

    // ==================== Process Last Tile (Computation only) ====================
    int curr_idx = (num_k_tiles - 1) & 1;

    // Load Last A
    wmma::load_matrix_sync(frag_a[0][0], &s_a[OFFSET(comp_c_frag_m * 64, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[0][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 0, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][0], &s_a[OFFSET(comp_c_frag_m * 64, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][1], &s_a[OFFSET(comp_c_frag_m * 64 + 16, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][2], &s_a[OFFSET(comp_c_frag_m * 64 + 32, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);
    wmma::load_matrix_sync(frag_a[1][3], &s_a[OFFSET(comp_c_frag_m * 64 + 48, 16, BK + APAD) + curr_idx * s_a_db_offset], BK + APAD);

    // Load Last B
    wmma::load_matrix_sync(frag_b[0][0], &s_b[OFFSET(comp_c_frag_n * 64, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[0][1], &s_b[OFFSET(comp_c_frag_n * 64 + 16, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[0][2], &s_b[OFFSET(comp_c_frag_n * 64 + 32, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[0][3], &s_b[OFFSET(comp_c_frag_n * 64 + 48, 0, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][0], &s_b[OFFSET(comp_c_frag_n * 64, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][1], &s_b[OFFSET(comp_c_frag_n * 64 + 16, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][2], &s_b[OFFSET(comp_c_frag_n * 64 + 32, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);
    wmma::load_matrix_sync(frag_b[1][3], &s_b[OFFSET(comp_c_frag_n * 64 + 48, 16, BK + BPAD) + curr_idx * s_b_db_offset], BK + BPAD);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
            wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
        }
    }

    // ==================== Store Result ====================
    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;

    float *s_c_float = reinterpret_cast<float *>(s_b + 2 * BN * (BK + BPAD));
    // Shared Memory buffer for BF16 output
    cuda_bfloat16 *s_c_bf16 = reinterpret_cast<cuda_bfloat16 *>(s_c_float + 8 * 16 * 16);

    // Preload Bias
    float bias_vals[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int global_n = store_c_gmem_n + j * 16 + (lane_id & 15);
        bias_vals[j] = (global_n < N && bias) ? __bfloat162float(bias[global_n]) : 0.0f;
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int tile_m = store_c_gmem_m + i * 16;
            int tile_n = store_c_gmem_n + j * 16;

            // Store accumulator to Shared Memory (float)
            wmma::store_matrix_sync(&s_c_float[wid * 256], frag_c[i][j], 16, wmma::mem_row_major);
            __syncwarp();

            // Add Bias & Convert to BF16
            #pragma unroll
            for (int idx = lane_id; idx < 256; idx += 32) {
                int local_m = idx >> 4;
                int local_n = idx & 15;
                
                float val = s_c_float[wid * 256 + local_m * 16 + local_n] + bias_vals[j];
                s_c_bf16[wid * 256 + local_m * 16 + local_n] = __float2bfloat16(val);
            }
            __syncwarp();

            // Write to Global Memory
            int row = lane_id >> 1;
            int col = (lane_id & 1) << 3;

            int global_m = tile_m + row;
            int global_n = tile_n + col;

            if (global_m < M && global_n + 7 < N) {
                // Vectorized store: 128-bit (8 x bf16)
                // Need a way to store 128 bits. int4 is convenient.
                int4 v;
                // Reinterpret cast from bf16* to int4* requires care, using memcpy or assignment
                // Assuming s_c_bf16 is aligned properly in SMEM
                v = *reinterpret_cast<int4*>(&s_c_bf16[wid * 256 + row * 16 + col]);
                
                reinterpret_cast<int4*>(&C[OFFSET(global_m, global_n, N)])[0] = v;
            } else if (global_m < M) {
                #pragma unroll
                for (int c = 0; c < 8; c++) {
                    if (global_n + c < N) {
                        C[OFFSET(global_m, global_n + c, N)] = s_c_bf16[wid * 256 + row * 16 + col + c];
                    }
                }
            }
            __syncwarp();
        }
    }
}