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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
	const bool debug = false)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kHalf);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<__half>(),
		W, H,
		means3D.contiguous().data<__half>(),
		sh.contiguous().data_ptr<__half>(),
		colors.contiguous().data<__half>(), 
		opacity.contiguous().data<__half>(), 
		scales.contiguous().data_ptr<__half>(),
		scale_modifier,
		rotations.contiguous().data_ptr<__half>(),
		cov3D_precomp.contiguous().data<__half>(), 
		viewmatrix.contiguous().data<__half>(), 
		projmatrix.contiguous().data<__half>(),
		campos.contiguous().data<__half>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<__half>(),
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

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
	const int degree =0,
	const torch::Tensor& campos = torch::empty({0}, torch::kHalf),
	const torch::Tensor& geomBuffer = torch::empty({0}, torch::kHalf),
	const int R = 0,
	const torch::Tensor& binningBuffer = torch::empty({0}, torch::kHalf),
	const torch::Tensor& imageBuffer = torch::empty({0}, torch::kHalf),
	const bool debug=false) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<__half>(),
	  W, H, 
	  means3D.contiguous().data<__half>(),		//
	  sh.contiguous().data<__half>(),
	  colors.contiguous().data<__half>(),
	  scales.data_ptr<__half>(),
	  scale_modifier,
	  rotations.data_ptr<__half>(),
	  cov3D_precomp.contiguous().data<__half>(),
	  viewmatrix.contiguous().data<__half>(),		//
	  projmatrix.contiguous().data<__half>(),
	  campos.contiguous().data<__half>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<__half>(),
	  dL_dmeans2D.contiguous().data<__half>(),
	  dL_dconic.contiguous().data<__half>(),  
	  dL_dopacity.contiguous().data<__half>(),
	  dL_dcolors.contiguous().data<__half>(),
	  dL_dmeans3D.contiguous().data<__half>(),
	  dL_dcov3D.contiguous().data<__half>(),
	  dL_dsh.contiguous().data<__half>(),
	  dL_dscales.contiguous().data<__half>(),
	  dL_drotations.contiguous().data<__half>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<__half>(),
		viewmatrix.contiguous().data<__half>(),
		projmatrix.contiguous().data<__half>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}