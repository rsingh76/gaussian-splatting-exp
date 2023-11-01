/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 means, glm::vec3 campos, const __half* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means;
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);
	int idx3 = idx * 3*max_coeffs;
	glm::vec3 sh_0 = { __half2float(shs[idx3]), __half2float(shs[idx3 + 1]),	__half2float(shs[idx3 + 2]) };
	glm::vec3 sh_1 = { __half2float(shs[idx3+3]), __half2float(shs[idx3+3 + 1]), __half2float(shs[idx3+3 + 2]) };
	glm::vec3 sh_2 = { __half2float(shs[idx3+6]), __half2float(shs[idx3+6 + 1]), __half2float(shs[idx3+6 + 2]) };
	glm::vec3 sh_3 = { __half2float(shs[idx3+9]), __half2float(shs[idx3+9 + 1]), __half2float(shs[idx3+9 + 2]) };
	glm::vec3 sh_4 = { __half2float(shs[idx3+12]), __half2float(shs[idx3+12 + 1]), __half2float(shs[idx3+12 + 2]) };
	glm::vec3 sh_5 = { __half2float(shs[idx3+15]), __half2float(shs[idx3+15 + 1]), __half2float(shs[idx3+15 + 2]) };
	glm::vec3 sh_6 = { __half2float(shs[idx3+18]), __half2float(shs[idx3+18 + 1]), __half2float(shs[idx3+18 + 2]) };
	glm::vec3 sh_7 = { __half2float(shs[idx3+21]), __half2float(shs[idx3+21 + 1]), __half2float(shs[idx3+21 + 2]) };
	glm::vec3 sh_8 = { __half2float(shs[idx3+24]), __half2float(shs[idx3+24 + 1]), __half2float(shs[idx3+24 + 2]) };
	glm::vec3 sh_9 = { __half2float(shs[idx3+27]), __half2float(shs[idx3+27 + 1]), __half2float(shs[idx3+27 + 2]) };
	glm::vec3 sh_10 = { __half2float(shs[idx3+30]), __half2float(shs[idx3+30 + 1]), __half2float(shs[idx3+30 + 2]) };
	glm::vec3 sh_11 = { __half2float(shs[idx3+33]), __half2float(shs[idx3+33 + 1]), __half2float(shs[idx3+33 + 2]) };
	glm::vec3 sh_12 = { __half2float(shs[idx3+36]), __half2float(shs[idx3+36 + 1]), __half2float(shs[idx3+36 + 2]) };
	glm::vec3 sh_13 = { __half2float(shs[idx3+39]), __half2float(shs[idx3+39 + 1]), __half2float(shs[idx3+39 + 2]) };
	glm::vec3 sh_14 = { __half2float(shs[idx3+42]), __half2float(shs[idx3+42 + 1]), __half2float(shs[idx3+42 + 2]) };
	glm::vec3 sh_15 = { __half2float(shs[idx3+45]), __half2float(shs[idx3+45 + 1]), __half2float(shs[idx3+45 + 2]) };

	// glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh_0;

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh_1 + SH_C1 * z * sh_2 - SH_C1 * x * sh_3;

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh_4 +
				SH_C2[1] * yz * sh_5 +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh_6 +
				SH_C2[3] * xz * sh_7 +
				SH_C2[4] * (xx - yy) * sh_8;

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh_9 +
					SH_C3[1] * xy * z * sh_10 +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh_11 +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh_12 +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh_13 +
					SH_C3[5] * z * (xx - yy) * sh_14 +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh_15;
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ __half3 computeCov2D(const __half3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const __half* cov3D, const __half* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		__half2float(viewmatrix[0]), __half2float(viewmatrix[4]), __half2float(viewmatrix[8]),
		__half2float(viewmatrix[1]), __half2float(viewmatrix[5]), __half2float(viewmatrix[9]),
		__half2float(viewmatrix[2]), __half2float(viewmatrix[6]), __half2float(viewmatrix[10]));

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		__half2float(cov3D[0]), __half2float(cov3D[1]), __half2float(cov3D[2]),
		__half2float(cov3D[1]), __half2float(cov3D[3]), __half2float(cov3D[4]),
		__half2float(cov3D[2]), __half2float(cov3D[4]), __half2float(cov3D[5]));

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { __float2half(cov[0][0]), __float2half(cov[0][1]), __float2half(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, __half* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = __float2half(Sigma[0][0]);
	cov3D[1] = __float2half(Sigma[0][1]);
	cov3D[2] = __float2half(Sigma[0][2]);
	cov3D[3] = __float2half(Sigma[1][1]);
	cov3D[4] = __float2half(Sigma[1][2]);
	cov3D[5] = __float2half(Sigma[2][2]);
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const __half* orig_points,
	const __half3* scales,
	const float scale_modifier,
	const __half4* rotations,
	const __half* opacities,
	const __half* shs,
	bool* clamped,
	const __half* cov3D_precomp,
	const __half* colors_precomp,
	const __half* viewmatrix,
	const __half* projmatrix,
	const __half3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	__half2* points_xy_image,
	__half* depths,
	__half* cov3Ds,
	__half* rgb,
	__half4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	__half3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const __half* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		glm::vec3 scale = { __half2float(scales[idx].x), __half2float(scales[idx].y), __half2float(scales[idx].z) };
		glm::vec4 rotation = { __half2float(rotations[idx].x), __half2float(rotations[idx].y), __half2float(rotations[idx].z), __half2float(rotations[idx].w) };
		computeCov3D(scale, scale_modifier, rotation, cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	__half3 cov_half = computeCov2D(p_orig, focal_x,focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);
	float3 cov = cov_half.convert();

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		// create an indexing that deivides orig_points into clusters of 3
		int idx3 = idx * 3;
		// create a float point array from orig_points with 3 elements starting with idx3 without using glm
		glm::vec3 p_orig_vec3 = { __half2float(orig_points[idx3]), __half2float(orig_points[idx3 + 1]), __half2float(orig_points[idx3 + 2]) };
	
		glm::vec3 campos_vec = { __half2float(cam_pos->x), __half2float(cam_pos->y), __half2float(cam_pos->z)};
		glm::vec3 result = computeColorFromSH(idx, D, M, p_orig_vec3, campos_vec, shs, clamped);
		rgb[idx * C + 0] = __float2half(result.x);
		rgb[idx * C + 1] = __float2half(result.y);
		rgb[idx * C + 2] = __float2half(result.z);
	}

	// Store some useful helper data for the next steps.
	depths[idx] = __float2half(p_view.z);
	radii[idx] = my_radius;
	points_xy_image[idx] = __float22half2(point_image);
	// Inverse 2D covariance and opacity neatly pack into one float4
	__half3 conic_half = __float32half3(conic);
	conic_opacity[idx] = { conic_half.x, conic_half.y, conic_half.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const __half2* __restrict__ points_xy_image,
	const __half* __restrict__ features,
	const __half4* __restrict__ conic_opacity,
	__half* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const __half* __restrict__ bg_color,
	__half* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	__half2 pixf = { __float2half((float)pix.x), __float2half((float)pix.y) };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ __half2 collected_xy[BLOCK_SIZE];
	__shared__ __half4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	__half T = __float2half(1.0f);
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	__half C[CHANNELS] = { __int2half_rn(0) };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			__half2 xy = collected_xy[j];
			__half2 d = {__hsub(xy.x , pixf.x), __hsub(xy.y , pixf.y) };
			__half4 con_o = collected_conic_opacity[j];
			__half power = __hsub(__hmul(__float2half(-0.5f) , __hadd(__hmul(con_o.x , __hmul(d.x , d.x)) , __hmul(con_o.z , __hmul(d.y , d.y)))) , __hmul(con_o.y , __hmul(d.x , d.y)));
			if (__hgt(power , __float2half(0.0f)))
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			__half alpha = __hmin(__float2half(0.99f), __hmul(con_o.w , hexp(power)));
			if (__hlt(alpha , __float2half(1.0f / 255.0f)))
				continue;
			__half test_T = __hmul(T , __hsub(__int2half_rn(1) , alpha));
			if (__hlt(test_T , __float2half(0.0001f)))
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] = __hadd(C[ch] , __hmul(features[collected_id[j] * CHANNELS + ch] , __hmul(alpha , T)));

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = __hadd(C[ch] , __hmul(T , bg_color[ch]));
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const __half2* means2D,
	const __half* colors,
	const __half4* conic_opacity,
	__half* final_T,
	uint32_t* n_contrib,
	const __half* bg_color,
	__half* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const __half* means3D,
	const __half3* scales,
	const float scale_modifier,
	const __half4* rotations,
	const __half* opacities,
	const __half* shs,
	bool* clamped,
	const __half* cov3D_precomp,
	const __half* colors_precomp,
	const __half* viewmatrix,
	const __half* projmatrix,
	const __half3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	__half2* means2D,
	__half* depths,
	__half* cov3Ds,
	__half* rgb,
	__half4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}