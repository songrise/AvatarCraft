#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include <cstdio>
#include <stdint.h>
#include <stdexcept>

#include "pcg32.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

// some const
inline constexpr __device__ float DENSITY_THRESH() { return 10.0f; } // TODO: how to decide this threshold ?
inline constexpr __device__ float SQRT3() { return 1.73205080757f; }
inline constexpr __device__ int MAX_STEPS() { return 1024; }
inline constexpr __device__ float MIN_STEPSIZE() { return 2 * SQRT3() / MAX_STEPS(); } // still need to mul bound to get dt_min
inline constexpr __device__ float MIN_NEAR() { return 0.05f; }
inline constexpr __device__ float DT_GAMMA() { return 1.f / 256.f; }

// util functions
template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

inline __host__ __device__ float signf(float x) {
	return copysignf(1.0, x);
}

inline __host__ __device__ float clamp(float x, const float min, const float max) {
    return fminf(max, fmaxf(min, x));
}

inline __host__ __device__ void swapf(float& a, float& b) {
	float c = a; a = b; b = c;
}

////////////////////////////////////////////////////
/////////////         training         /////////////
////////////////////////////////////////////////////

// rays_o/d: [N, 3]
// grid: [H, H, H]
// xyzs, dirs, deltas: [M, 3], [M, 3], [M]
// dirs: [M, 3]
// rays: [N, 3], idx, offset, num_steps
template <typename scalar_t>
__global__ void kernel_march_rays_train(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,  
    const scalar_t * __restrict__ grid,
    const float mean_density,
    const int iter_density,
    const float bound,
    const uint32_t N, const uint32_t H, const uint32_t M,
    scalar_t * xyzs, scalar_t * dirs, scalar_t * deltas,
    int * rays,
    int * counter,
    const uint32_t perturb
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    const float rbound = 1 / bound;

    const float density_thresh = fminf(DENSITY_THRESH(), mean_density);

    // locate
    rays_o += n * 3;
    rays_d += n * 3;

    // ray marching (naive, no mip, just one level)
    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // get near far (assume cube scene)
    float near_x = (-bound - ox) * rdx;
    float far_x = (bound - ox) * rdx;
    if (near_x > far_x) swapf(near_x, far_x);
    float near_y = (-bound - oy) * rdy;
    float far_y = (bound - oy) * rdy;
    if (near_y > far_y) swapf(near_y, far_y);
    float near_z = (-bound - oz) * rdz;
    float far_z = (bound - oz) * rdz;
    if (near_z > far_z) swapf(near_z, far_z);

    const float near = fmaxf(fmaxf(near_x, fmaxf(near_y, near_z)), MIN_NEAR()); // hardcoded minimal near distance
    const float far = fminf(far_x, fminf(far_y, far_z));

    const float dt_min = MIN_STEPSIZE() * bound;
    const float dt_max = 2 * bound / (H - 1);
    const float dt_gamma = bound > 1 ? DT_GAMMA() : 0.0f;

    float t0 = near;
    if (perturb) {
        pcg32 rng((uint64_t)n);
        t0 += dt_min * rng.next_float();
    }

    // first pass: estimation of num_steps
    float t = t0;
    uint32_t num_steps = 0;

    //if (t < far) printf("valid ray %d t=%f near=%f far=%f \n", n, t, near, far);

    while (t < far && num_steps < MAX_STEPS()) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * rbound + 1) * H, 0.0f, (float)(H - 1));

        const uint32_t index = nx * H * H + ny * H + nz;
        const float density = grid[index];

        // if occpuied, advance a small step, and write to output
        //if (n == 0) printf("t=%f density=%f vs thresh=%f step=%d\n", t, density, density_thresh, num_steps);

        if (density > density_thresh) {
            num_steps++;
            const float dt = clamp(t * dt_gamma, dt_min, dt_max);
            t += dt;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) / (H - 1) * 2 - 1) * bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) / (H - 1) * 2 - 1) * bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) / (H - 1) * 2 - 1) * bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                const float dt = clamp(t * dt_gamma, dt_min, dt_max);
                t += dt; 
            } while (t < tt);
        }
    }

    //printf("[n=%d] num_steps=%d\n", n, num_steps);
    //printf("[n=%d] num_steps=%d, pc=%d, rc=%d\n", n, num_steps, counter[0], counter[1]);


    // second pass: really locate and write points & dirs
    uint32_t point_index = atomicAdd(counter, num_steps);
    uint32_t ray_index = atomicAdd(counter + 1, 1);
    
    //printf("[n=%d] num_steps=%d, point_index=%d, ray_index=%d\n", n, num_steps, point_index, ray_index);

    // write rays
    rays[ray_index * 3] = n;
    rays[ray_index * 3 + 1] = point_index;
    rays[ray_index * 3 + 2] = num_steps;

    if (num_steps == 0) return;
    if (point_index + num_steps >= M) return;

    xyzs += point_index * 3;
    dirs += point_index * 3;
    deltas += point_index;

    t = t0;
    uint32_t step = 0;

    while (t < far && step < num_steps) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * rbound + 1) * H, 0.0f, (float)(H - 1));

        // query grid
        const uint32_t index = nx * H * H + ny * H + nz;
        const float density = grid[index];

        // if occpuied, advance a small step, and write to output
        if (density > density_thresh) {
            // write step
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;
            const float dt = clamp(t * dt_gamma, dt_min, dt_max);
            t += dt;
            deltas[0] = dt;
            xyzs += 3;
            dirs += 3;
            deltas++;
            step++;
        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) / (H - 1) * 2 - 1) * bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) / (H - 1) * 2 - 1) * bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) / (H - 1) * 2 - 1) * bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                const float dt = clamp(t * dt_gamma, dt_min, dt_max);
                t += dt; 
            } while (t < tt);
        }
    }
}


// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N], final pixel alpha
// image: [N, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_forward(
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,  
    const scalar_t * __restrict__ deltas,
    const int * __restrict__ rays,
    const float bound,
    const uint32_t M, const uint32_t N,
    scalar_t * weights_sum, // temp: used as weights_sum
    scalar_t * image
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    // empty ray, or ray that exceed max step count.
    if (num_steps == 0 || offset + num_steps >= M) {
        weights_sum[index] = 0;
        image[index * 3] = 0;
        image[index * 3 + 1] = 0;
        image[index * 3 + 2] = 0;
        return;
    }

    sigmas += offset;
    rgbs += offset * 3;
    deltas += offset;

    // accumulate 
    uint32_t step = 0;
    scalar_t T = 1.0f;

    scalar_t r = 0, g = 0, b = 0, d = 0;

    while (step < num_steps) {

        // minimal remained transmittence
        if (T < 1e-4f) break;

        const scalar_t alpha =  sigmas[0]; //1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;

        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];

        T *= 1.0f - alpha;

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // locate
        sigmas++;
        rgbs += 3;
        deltas++;

        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // write
    weights_sum[index] = 1.0f - T; // weights_sum
    image[index * 3] = r;
    image[index * 3 + 1] = g;
    image[index * 3 + 2] = b;
}


// grad_weights_sum: [N,]
// grad: [N, 3]
// sigmas: [M]
// rgbs: [M, 3]
// deltas: [M]
// rays: [N, 3], idx, offset, num_steps
// weights_sum: [N,], weights_sum here 
// image: [N, 3]
// grad_sigmas: [M]
// grad_rgbs: [M, 3]
template <typename scalar_t>
__global__ void kernel_composite_rays_train_backward(
    const scalar_t * __restrict__ grad_weights_sum,
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ sigmas,
    const scalar_t * __restrict__ rgbs,  
    const scalar_t * __restrict__ deltas,  
    const int * __restrict__ rays,
    const scalar_t * __restrict__ weights_sum,
    const scalar_t * __restrict__ image,  
    const float bound,
    const uint32_t M, const uint32_t N,
    scalar_t * grad_sigmas,
    scalar_t * grad_rgbs
) {
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate 
    uint32_t index = rays[n * 3];
    uint32_t offset = rays[n * 3 + 1];
    uint32_t num_steps = rays[n * 3 + 2];

    if (num_steps == 0 || offset + num_steps >= M) return;

    grad_weights_sum += index;
    grad += index * 3;
    weights_sum += index;
    image += index * 3;
    sigmas += offset;
    rgbs += offset * 3;
    deltas += offset;
    grad_sigmas += offset;
    grad_rgbs += offset * 3;

    // accumulate 
    uint32_t step = 0;
    scalar_t T = 1.0f;

    const scalar_t r_final = image[0], g_final = image[1], b_final = image[2], T_final = 1 - weights_sum[0];
    scalar_t r = 0, g = 0, b = 0;

    while (step < num_steps) {
        
        //if (T < 1e-4f) break;

        const scalar_t alpha =  sigmas[0]; //1.0f - __expf(- sigmas[0] * deltas[0]);
        const scalar_t weight = alpha * T;

        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];

        T *= 1.0f - alpha; // this has been T(t+1)

        // write grad
        grad_rgbs[0] = grad[0] * weight;
        grad_rgbs[1] = grad[1] * weight;
        grad_rgbs[2] = grad[2] * weight;

        grad_sigmas[0] = deltas[0] * (
            grad[0] * (T * rgbs[0] - (r_final - r)) + 
            grad[1] * (T * rgbs[1] - (g_final - g)) + 
            grad[2] * (T * rgbs[2] - (b_final - b)) +
            grad_weights_sum[0] * T_final
        );
    
        // locate
        sigmas++;
        rgbs += 3;
        grad_sigmas++;
        grad_rgbs += 3;
        deltas++;

        step++;
    }
}


void march_rays_train(at::Tensor rays_o, at::Tensor rays_d, at::Tensor grid, const float mean_density, const int iter_density, const float bound, const uint32_t N, const uint32_t H, const uint32_t M, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter, const uint32_t perturb) {
    CHECK_CUDA(rays_o);
    CHECK_CUDA(rays_d);
    CHECK_CUDA(grid);

    CHECK_CONTIGUOUS(rays_o);
    CHECK_CONTIGUOUS(rays_d);
    CHECK_CONTIGUOUS(grid);

    CHECK_IS_FLOATING(rays_o);
    CHECK_IS_FLOATING(rays_d);
    CHECK_IS_FLOATING(grid);

    static constexpr uint32_t N_THREAD = 256;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays_train", ([&] {
        kernel_march_rays_train<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), grid.data_ptr<scalar_t>(), mean_density, iter_density, bound, N, H, M, xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), counter.data_ptr<int>(), perturb);
    }));
}


void composite_rays_train_forward(at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor rays, const float bound, const uint32_t M, const uint32_t N, at::Tensor weights_sum, at::Tensor image) {

    CHECK_CUDA(sigmas);
    CHECK_CUDA(rgbs);
    CHECK_CUDA(deltas);
    CHECK_CUDA(rays);
    CHECK_CUDA(weights_sum);
    CHECK_CUDA(image);

    CHECK_CONTIGUOUS(sigmas);
    CHECK_CONTIGUOUS(rgbs);
    CHECK_CONTIGUOUS(deltas);
    CHECK_CONTIGUOUS(rays);
    CHECK_CONTIGUOUS(weights_sum);
    CHECK_CONTIGUOUS(image);

    CHECK_IS_FLOATING(sigmas);
    CHECK_IS_FLOATING(rgbs);
    CHECK_IS_FLOATING(deltas);
    CHECK_IS_INT(rays);
    CHECK_IS_FLOATING(weights_sum);
    CHECK_IS_FLOATING(image);

    static constexpr uint32_t N_THREAD = 256;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sigmas.scalar_type(), "composite_rays_train_forward", ([&] {
        kernel_composite_rays_train_forward<<<div_round_up(N, N_THREAD), N_THREAD>>>(sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), bound, M, N, weights_sum.data_ptr<scalar_t>(), image.data_ptr<scalar_t>());
    }));
}


void composite_rays_train_backward(at::Tensor grad_weights_sum, at::Tensor grad, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor rays, at::Tensor weights_sum, at::Tensor image, const float bound, const uint32_t M, const uint32_t N, at::Tensor grad_sigmas, at::Tensor grad_rgbs) {
    
    CHECK_CUDA(grad_weights_sum);
    CHECK_CUDA(grad);
    CHECK_CUDA(sigmas);
    CHECK_CUDA(rgbs);
    CHECK_CUDA(deltas);
    CHECK_CUDA(rays);
    CHECK_CUDA(weights_sum);
    CHECK_CUDA(image);
    CHECK_CUDA(grad_sigmas);
    CHECK_CUDA(grad_rgbs);
    
    CHECK_CONTIGUOUS(grad_weights_sum);
    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(sigmas);
    CHECK_CONTIGUOUS(rgbs);
    CHECK_CONTIGUOUS(deltas);
    CHECK_CONTIGUOUS(rays);
    CHECK_CONTIGUOUS(weights_sum);
    CHECK_CONTIGUOUS(image);
    CHECK_CONTIGUOUS(grad_sigmas);
    CHECK_CONTIGUOUS(grad_rgbs);

    CHECK_IS_FLOATING(grad_weights_sum);
    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(sigmas);
    CHECK_IS_FLOATING(rgbs);
    CHECK_IS_FLOATING(deltas);
    CHECK_IS_INT(rays);
    CHECK_IS_FLOATING(weights_sum);
    CHECK_IS_FLOATING(image);
    CHECK_IS_FLOATING(grad_sigmas);
    CHECK_IS_FLOATING(grad_rgbs);

    static constexpr uint32_t N_THREAD = 256;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "composite_rays_train_backward", ([&] {
        kernel_composite_rays_train_backward<<<div_round_up(N, N_THREAD), N_THREAD>>>(grad_weights_sum.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), weights_sum.data_ptr<scalar_t>(), image.data_ptr<scalar_t>(), bound, M, N, grad_sigmas.data_ptr<scalar_t>(), grad_rgbs.data_ptr<scalar_t>());
    }));
}


////////////////////////////////////////////////////
/////////////          infernce        /////////////
////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void kernel_march_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const int* __restrict__ rays_alive, 
    const scalar_t* __restrict__ rays_t, 
    const scalar_t* __restrict__ rays_o, 
    const scalar_t* __restrict__ rays_d, 
    const float bound,
    const uint32_t H,
    const scalar_t * __restrict__ grid,
    const float mean_density,
    const scalar_t* __restrict__ nears,
    const scalar_t* __restrict__ fars,
    scalar_t* xyzs, scalar_t* dirs, scalar_t* deltas,
    const uint32_t perturb
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    float t = rays_t[n]; // current ray's t

    const float rbound = 1 / bound;
    const float density_thresh = fminf(DENSITY_THRESH(), mean_density);

    // locate
    rays_o += index * 3;
    rays_d += index * 3;
    xyzs += n * n_step * 3;
    dirs += n * n_step * 3;
    deltas += n * n_step * 2;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;
    const float near = nears[index], far = fars[index];

    const float dt_min = MIN_STEPSIZE() * bound;
    const float dt_max = 2 * bound / (H - 1);
    const float dt_gamma = bound > 1 ? DT_GAMMA() : 0.0f;

    // march for n_step steps, record points
    uint32_t step = 0;

    // introduce some randomness (pass in spp as perturb here)
    if (perturb) {
        pcg32 rng((uint64_t)n, (uint64_t)perturb);
        t += dt_min * rng.next_float();
    }

    float last_t = t;

    while (t < far && step < n_step) {
        // current point
        const float x = clamp(ox + t * dx, -bound, bound);
        const float y = clamp(oy + t * dy, -bound, bound);
        const float z = clamp(oz + t * dz, -bound, bound);

        // convert to nearest grid position
        const int nx = clamp(0.5 * (x * rbound + 1) * H, 0.0f, (float)(H - 1));
        const int ny = clamp(0.5 * (y * rbound + 1) * H, 0.0f, (float)(H - 1));
        const int nz = clamp(0.5 * (z * rbound + 1) * H, 0.0f, (float)(H - 1));

        // query grid
        const uint32_t index = nx * H * H + ny * H + nz;
        const float density = grid[index];

        // if occpuied, advance a small step, and write to output
        if (density > density_thresh) {
            // write step
            xyzs[0] = x;
            xyzs[1] = y;
            xyzs[2] = z;
            dirs[0] = dx;
            dirs[1] = dy;
            dirs[2] = dz;
            // calc dt
            const float dt = clamp(t * dt_gamma, dt_min, dt_max);
            t += dt;
            deltas[0] = dt;
            deltas[1] = t - last_t; // used to calc depth
            last_t = t;
            // step
            xyzs += 3;
            dirs += 3;
            deltas += 2;
            step++;

        // else, skip a large step (basically skip a voxel grid)
        } else {
            // calc distance to next voxel
            const float tx = (((nx + 0.5f + 0.5f * signf(dx)) / (H - 1) * 2 - 1) * bound - x) * rdx;
            const float ty = (((ny + 0.5f + 0.5f * signf(dy)) / (H - 1) * 2 - 1) * bound - y) * rdy;
            const float tz = (((nz + 0.5f + 0.5f * signf(dz)) / (H - 1) * 2 - 1) * bound - z) * rdz;
            const float tt = t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                const float dt = clamp(t * dt_gamma, dt_min, dt_max);
                t += dt; 
            } while (t < tt);
        }
    }
}

void march_rays(const uint32_t n_alive, const uint32_t n_step, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor rays_o, at::Tensor rays_d, const float bound, const uint32_t H, at::Tensor density_grid, const float mean_density, at::Tensor near, at::Tensor far, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, const uint32_t perturb) {
    static constexpr uint32_t N_THREAD = 256;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays", ([&] {
        kernel_march_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), bound, H, density_grid.data_ptr<scalar_t>(), mean_density, near.data_ptr<scalar_t>(), far.data_ptr<scalar_t>(), xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), perturb);
    }));
}


template <typename scalar_t>
__global__ void kernel_composite_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const int* __restrict__ rays_alive, 
    scalar_t* rays_t, 
    const scalar_t* __restrict__ sigmas, 
    const scalar_t* __restrict__ rgbs, 
    const scalar_t* __restrict__ normals,
    const scalar_t* __restrict__ deltas, 
    scalar_t* weights_sum, scalar_t* depth, scalar_t* image, scalar_t* normal_map
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    scalar_t t = rays_t[n]; // current ray's t

    // locate 
    sigmas += n * n_step;
    rgbs += n * n_step * 3;
    deltas += n * n_step * 2;
    normals += n * n_step * 3;


    weights_sum += index;
    depth += index;
    image += index * 3;
    normal_map += index * 3;
    
    scalar_t weight_sum = weights_sum[0];
    scalar_t d = depth[0];
    scalar_t r = image[0];
    scalar_t g = image[1];
    scalar_t b = image[2];
    scalar_t nx = normal_map[0];
    scalar_t ny = normal_map[1];
    scalar_t nz = normal_map[2];

    // accumulate 
    uint32_t step = 0;
    while (step < n_step) {
        
        // ray is terminated if delta == 0
        if (deltas[0] == 0) break;
        
        const scalar_t alpha = sigmas[0];  //1.0f - __expf(- sigmas[0] * deltas[0]);

        /* 
        T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
        w_i = alpha_i * T_i
        --> 
        T_i = 1 - \sum_{j=0}^{i-1} w_j
        */
        const scalar_t T = 1 - weight_sum;
        const scalar_t weight = alpha * T;
        weight_sum += weight;

        t += deltas[1]; // real delta
        d += weight * t;
        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];
        nx += weight * normals[0];
        ny += weight * normals[1];
        nz += weight * normals[2];

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // ray is terminated if T is too small
        if (T < 1e-2) break;

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;
        normals+=3;
        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // rays_t = -1 means ray is terminated early.
    if (step < n_step) {
        rays_t[n] = -1;
    } else {
        rays_t[n] = t;
    }

    weights_sum[0] = weight_sum;
    depth[0] = d;
    image[0] = r;
    image[1] = g;
    image[2] = b;
    normal_map[0] = nx;
    normal_map[1] = ny;
    normal_map[2] = nz;
}


void composite_rays(const uint32_t n_alive, const uint32_t n_step, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor normals, at::Tensor deltas, at::Tensor weights, at::Tensor depth, at::Tensor image, at::Tensor normal_map) {
    static constexpr uint32_t N_THREAD = 256;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    sigmas.scalar_type(), "composite_rays", ([&] {
        kernel_composite_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, 
                                                                            rays_alive.data_ptr<int>(), 
                                                                            rays_t.data_ptr<scalar_t>(), 
                                                                            sigmas.data_ptr<scalar_t>(), 
                                                                            rgbs.data_ptr<scalar_t>(), 
                                                                            normals.data_ptr<scalar_t>(),
                                                                            deltas.data_ptr<scalar_t>(), 
                                                                            weights.data_ptr<scalar_t>(), 
                                                                            depth.data_ptr<scalar_t>(), 
                                                                            image.data_ptr<scalar_t>(),
                                                                            normal_map.data_ptr<scalar_t>());
    }));
}


template <typename scalar_t>
__global__ void kernel_compact_rays(
    const uint32_t n_alive, 
    int* rays_alive, 
    const int* __restrict__ rays_alive_old, 
    scalar_t* rays_t, 
    const scalar_t* __restrict__ rays_t_old, 
    int* alive_counter
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    // rays_t_old[n] < 0 means ray died in last composite kernel.
    if (rays_t_old[n] >= 0) {
        const int index = atomicAdd(alive_counter, 1);
        rays_alive[index] = rays_alive_old[n];
        rays_t[index] = rays_t_old[n];
    }
}


void compact_rays(const uint32_t n_alive, at::Tensor rays_alive, at::Tensor rays_alive_old, at::Tensor rays_t, at::Tensor rays_t_old, at::Tensor alive_counter) {
    static constexpr uint32_t N_THREAD = 256;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_t.scalar_type(), "compact_rays", ([&] {
        kernel_compact_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, rays_alive.data_ptr<int>(), rays_alive_old.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), rays_t_old.data_ptr<scalar_t>(), alive_counter.data_ptr<int>());
    }));
}