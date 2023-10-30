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

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background = torch::empty({0}, torch::kHalf),
	const torch::Tensor& means3D = torch::empty({0}, torch::kHalf),
    const torch::Tensor& colors = torch::empty({0}, torch::kHalf),
    const torch::Tensor& opacity = torch::empty({0}, torch::kHalf),
	const torch::Tensor& scales = torch::empty({0}, torch::kHalf),
	const torch::Tensor& rotations = torch::empty({0}, torch::kHalf),
	const float scale_modifier = 0.0,
	const torch::Tensor& cov3D_precomp = torch::empty({0}, torch::kHalf),
	const torch::Tensor& viewmatrix = torch::empty({0}, torch::kHalf),
	const torch::Tensor& projmatrix = torch::empty({0}, torch::kHalf),
	const float tan_fovx = 0.0, 
	const float tan_fovy = 0.0,
    const int image_height = 0,
    const int image_width = 0,
	const torch::Tensor& sh = torch::empty({0}, torch::kHalf),
	const int degree = 0,
	const torch::Tensor& campos = torch::empty({0}, torch::kHalf),
	const bool prefiltered = false,
	const bool debug = false);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background = torch::empty({0}, torch::kHalf),
	const torch::Tensor& means3D = torch::empty({0}, torch::kHalf),
	const torch::Tensor& radii = torch::empty({0}, torch::kHalf),
    const torch::Tensor& colors = torch::empty({0}, torch::kHalf),
	const torch::Tensor& scales = torch::empty({0}, torch::kHalf),
	const torch::Tensor& rotations = torch::empty({0}, torch::kHalf),
	const float scale_modifier = 0.0f,
	const torch::Tensor& cov3D_precomp = torch::empty({0}, torch::kHalf),
	const torch::Tensor& viewmatrix = torch::empty({0}, torch::kHalf),
    const torch::Tensor& projmatrix = torch::empty({0}, torch::kHalf),
	const float tan_fovx = 0.0f, 
	const float tan_fovy = 0.0f,
    const torch::Tensor& dL_dout_color = torch::empty({0}, torch::kHalf),
	const torch::Tensor& sh = torch::empty({0}, torch::kHalf),
	const int degree = 0,
	const torch::Tensor& campos = torch::empty({0}, torch::kHalf),
	const torch::Tensor& geomBuffer = torch::empty({0}, torch::kHalf),
	const int R = 0,
	const torch::Tensor& binningBuffer = torch::empty({0}, torch::kHalf),
	const torch::Tensor& imageBuffer = torch::empty({0}, torch::kHalf),
	const bool debug=false);
		
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix);