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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include "cuda_fp16.h"
namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			__half* means3D,
			__half* viewmatrix,
			__half* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, int D, int M,
			const __half* background,
			const int width, int height,
			const __half* means3D,
			const __half* shs,
			const __half* colors_precomp,
			const __half* opacities,
			const __half* scales,
			const float scale_modifier,
			const __half* rotations,
			const __half* cov3D_precomp,
			const __half* viewmatrix,
			const __half* projmatrix,
			const __half* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			__half* out_color,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R,
			const __half* background,
			const int width, int height,
			const __half* means3D,
			const __half* shs,
			const __half* colors_precomp,
			const __half* scales,
			const float scale_modifier,
			const __half* rotations,
			const __half* cov3D_precomp,
			const __half* viewmatrix,
			const __half* projmatrix,
			const __half* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const __half* dL_dpix,
			__half* dL_dmean2D,
			__half* dL_dconic,
			__half* dL_dopacity,
			__half* dL_dcolor,
			__half* dL_dmean3D,
			__half* dL_dcov3D,
			__half* dL_dsh,
			__half* dL_dscale,
			__half* dL_drot,
			bool debug);
	};
};

#endif