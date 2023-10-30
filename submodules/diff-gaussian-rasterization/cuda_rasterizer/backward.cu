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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3 means, glm::vec3 campos, const __half3* shs, const bool* clamped, const glm::vec3 dL_dcolor, __half3* dL_dmeans, __half3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means;
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	int idx_new = idx*max_coeffs;
	__half3* sh_h = dL_dshs + idx_new;

	glm::vec3 sh_0	= {__half2float(sh_h[0].x), __half2float(sh_h[0].y), __half2float(sh_h[0].z)};
	glm::vec3 sh_1	= {__half2float(sh_h[1].x), __half2float(sh_h[1].y), __half2float(sh_h[1].z)};
	glm::vec3 sh_2	= {__half2float(sh_h[2].x), __half2float(sh_h[2].y), __half2float(sh_h[2].z)};
	glm::vec3 sh_3	= {__half2float(sh_h[3].x), __half2float(sh_h[3].y), __half2float(sh_h[3].z)};
	glm::vec3 sh_4	= {__half2float(sh_h[4].x), __half2float(sh_h[4].y), __half2float(sh_h[4].z)};
	glm::vec3 sh_5	= {__half2float(sh_h[5].x), __half2float(sh_h[5].y), __half2float(sh_h[5].z)};
	glm::vec3 sh_6	= {__half2float(sh_h[6].x), __half2float(sh_h[6].y), __half2float(sh_h[6].z)};
	glm::vec3 sh_7	= {__half2float(sh_h[7].x), __half2float(sh_h[7].y), __half2float(sh_h[7].z)};
	glm::vec3 sh_8	= {__half2float(sh_h[8].x), __half2float(sh_h[8].y), __half2float(sh_h[8].z)};
	glm::vec3 sh_9	= {__half2float(sh_h[9].x), __half2float(sh_h[9].y), __half2float(sh_h[9].z)};
	glm::vec3 sh_10	= {__half2float(sh_h[10].x), __half2float(sh_h[10].y), __half2float(sh_h[10].z)};
	glm::vec3 sh_11	= {__half2float(sh_h[11].x), __half2float(sh_h[11].y), __half2float(sh_h[11].z)};
	glm::vec3 sh_12	= {__half2float(sh_h[12].x), __half2float(sh_h[12].y), __half2float(sh_h[12].z)};
	glm::vec3 sh_13	= {__half2float(sh_h[13].x), __half2float(sh_h[13].y), __half2float(sh_h[13].z)};
	glm::vec3 sh_14	= {__half2float(sh_h[14].x), __half2float(sh_h[14].y), __half2float(sh_h[14].z)};
	glm::vec3 sh_15	= {__half2float(sh_h[15].x), __half2float(sh_h[15].y), __half2float(sh_h[15].z)};



	// glm::vec3* sh = shs + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor;
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	__half3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = {__float2half(dRGBdsh0 * dL_dRGB.x), __float2half(dRGBdsh0 * dL_dRGB.y), __float2half(dRGBdsh0 * dL_dRGB.z)};
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = { __float2half(dRGBdsh1 * dL_dRGB.x), __float2half(dRGBdsh1 * dL_dRGB.y), __float2half(dRGBdsh1 * dL_dRGB.z) };
		dL_dsh[2] = { __float2half(dRGBdsh2 * dL_dRGB.x), __float2half(dRGBdsh2 * dL_dRGB.y), __float2half(dRGBdsh2 * dL_dRGB.z) };
		dL_dsh[3] = { __float2half(dRGBdsh3 * dL_dRGB.x), __float2half(dRGBdsh3 * dL_dRGB.y), __float2half(dRGBdsh3 * dL_dRGB.z) };

		dRGBdx = -SH_C1 * sh_3;
		dRGBdy = -SH_C1 * sh_1;
		dRGBdz = SH_C1 * sh_2;

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = { __float2half(dRGBdsh4 * dL_dRGB.x), __float2half(dRGBdsh4 * dL_dRGB.y), __float2half(dRGBdsh4 * dL_dRGB.z) };
			dL_dsh[5] = { __float2half(dRGBdsh5 * dL_dRGB.x), __float2half(dRGBdsh5 * dL_dRGB.y), __float2half(dRGBdsh5 * dL_dRGB.z) };
			dL_dsh[6] = { __float2half(dRGBdsh6 * dL_dRGB.x), __float2half(dRGBdsh6 * dL_dRGB.y), __float2half(dRGBdsh6 * dL_dRGB.z) };
			dL_dsh[7] = { __float2half(dRGBdsh7 * dL_dRGB.x), __float2half(dRGBdsh7 * dL_dRGB.y), __float2half(dRGBdsh7 * dL_dRGB.z) };
			dL_dsh[8] = { __float2half(dRGBdsh8 * dL_dRGB.x), __float2half(dRGBdsh8 * dL_dRGB.y), __float2half(dRGBdsh8 * dL_dRGB.z) };

			dRGBdx += SH_C2[0] * y * sh_4 + SH_C2[2] * 2.f * -x * sh_6 + SH_C2[3] * z * sh_7 + SH_C2[4] * 2.f * x * sh_8;
			dRGBdy += SH_C2[0] * x * sh_4 + SH_C2[1] * z * sh_5 + SH_C2[2] * 2.f * -y * sh_6 + SH_C2[4] * 2.f * -y * sh_8;
			dRGBdz += SH_C2[1] * y * sh_5 + SH_C2[2] * 2.f * 2.f * z * sh_6 + SH_C2[3] * x * sh_7;

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = { __float2half(dRGBdsh9 * dL_dRGB.x), __float2half(dRGBdsh9 * dL_dRGB.y), __float2half(dRGBdsh9 * dL_dRGB.z) };
				dL_dsh[10] = { __float2half(dRGBdsh10 * dL_dRGB.x), __float2half(dRGBdsh10 * dL_dRGB.y), __float2half(dRGBdsh10 * dL_dRGB.z) };
				dL_dsh[11] = { __float2half(dRGBdsh11 * dL_dRGB.x), __float2half(dRGBdsh11 * dL_dRGB.y), __float2half(dRGBdsh11 * dL_dRGB.z) };
				dL_dsh[12] = { __float2half(dRGBdsh12 * dL_dRGB.x), __float2half(dRGBdsh12 * dL_dRGB.y), __float2half(dRGBdsh12 * dL_dRGB.z) };
				dL_dsh[13] = { __float2half(dRGBdsh13 * dL_dRGB.x), __float2half(dRGBdsh13 * dL_dRGB.y), __float2half(dRGBdsh13 * dL_dRGB.z) };
				dL_dsh[14] = { __float2half(dRGBdsh14 * dL_dRGB.x), __float2half(dRGBdsh14 * dL_dRGB.y), __float2half(dRGBdsh14 * dL_dRGB.z) };
				dL_dsh[15] = { __float2half(dRGBdsh15 * dL_dRGB.x), __float2half(dRGBdsh15 * dL_dRGB.y), __float2half(dRGBdsh15 * dL_dRGB.z) };

				dRGBdx += (
					SH_C3[0] * sh_9 * 3.f * 2.f * xy +
					SH_C3[1] * sh_10 * yz +
					SH_C3[2] * sh_11 * -2.f * xy +
					SH_C3[3] * sh_12 * -3.f * 2.f * xz +
					SH_C3[4] * sh_13 * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh_14 * 2.f * xz +
					SH_C3[6] * sh_15 * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh_9 * 3.f * (xx - yy) +
					SH_C3[1] * sh_10 * xz +
					SH_C3[2] * sh_11 * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh_12 * -3.f * 2.f * yz +
					SH_C3[4] * sh_13 * -2.f * xy +
					SH_C3[5] * sh_14 * -2.f * yz +
					SH_C3[6] * sh_15 * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh_10 * xy +
					SH_C3[2] * sh_11 * 4.f * 2.f * yz +
					SH_C3[3] * sh_12 * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh_13 * 4.f * 2.f * xz +
					SH_C3[5] * sh_14 * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx].x = __hadd(dL_dmeans[idx].x , __float2half(dL_dmean.x));
	dL_dmeans[idx].y = __hadd(dL_dmeans[idx].y , __float2half(dL_dmean.y));
	dL_dmeans[idx].z = __hadd(dL_dmeans[idx].z , __float2half(dL_dmean.z));
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const __half3* means,
	const int* radii,
	const __half* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const __half* view_matrix,
	const __half* dL_dconics,
	__half3* dL_dmeans,
	__half* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian

	const __half* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	__half3 mean = means[idx];
	float3 dL_dconic = { __half2float(dL_dconics[4 * idx]), __half2float(dL_dconics[4 * idx + 1]), __half2float(dL_dconics[4 * idx + 3]) };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		__half2float(view_matrix[0]), __half2float(view_matrix[4]), __half2float(view_matrix[8]),
		__half2float(view_matrix[1]), __half2float(view_matrix[5]), __half2float(view_matrix[9]),
		__half2float(view_matrix[2]), __half2float(view_matrix[6]), __half2float(view_matrix[10]));

	glm::mat3 Vrk = glm::mat3(
		__half2float(cov3D[0]), __half2float(cov3D[1]), __half2float(cov3D[2]),
		__half2float(cov3D[1]), __half2float(cov3D[3]), __half2float(cov3D[4]),
		__half2float(cov3D[2]), __half2float(cov3D[4]), __half2float(cov3D[5]));

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = __float2half(T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = __float2half(T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = __float2half(T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = __float2half(2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 2] = __float2half(2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc);
		dL_dcov[6 * idx + 4] = __float2half(2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc);
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = __int2half_rn(0);
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = {__float2half(dL_dmean.x), __float2half(dL_dmean.y), __float2half(dL_dmean.z)};
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const __half* dL_dcov3Ds, __half3* dL_dscales, __half4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const __half* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(__half2float(dL_dcov3D[0]), __half2float(dL_dcov3D[3]), __half2float(dL_dcov3D[5]));
	glm::vec3 ounc = 0.5f * glm::vec3(__half2float(dL_dcov3D[1]), __half2float(dL_dcov3D[2]), __half2float(dL_dcov3D[4]));

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		__half2float(dL_dcov3D[0]), 0.5f * __half2float(dL_dcov3D[1]), 0.5f * __half2float(dL_dcov3D[2]),
		0.5f * __half2float(dL_dcov3D[1]), __half2float(dL_dcov3D[3]), 0.5f * __half2float(dL_dcov3D[4]),
		0.5f * __half2float(dL_dcov3D[2]), 0.5f * __half2float(dL_dcov3D[4]), __half2float(dL_dcov3D[5])
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	__half3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = __float2half(glm::dot(Rt[0], dL_dMt[0]));
	dL_dscale->y = __float2half(glm::dot(Rt[1], dL_dMt[1]));
	dL_dscale->z = __float2half(glm::dot(Rt[2], dL_dMt[2]));

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	__half4* dL_drot = dL_drots + idx;
	*dL_drot = __half4{ __float2half(dL_dq.x), __float2half(dL_dq.y), __float2half(dL_dq.z), __float2half(dL_dq.w) };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const __half3* means,		
	const int* radii,
	const __half* shs,
	const bool* clamped,
	const __half3* scales,	// used to be glm::vec3
	const __half4* rotations,	// used to be glm::vec3
	const float scale_modifier,
	const __half* proj,
	const __half3* campos,		// used to be glm::vec3
	const __half3* dL_dmean2D,
	__half3* dL_dmeans,			// used to be glm::vec3
	__half* dL_dcolor,
	__half* dL_dcov3D,
	__half* dL_dsh,
	__half3* dL_dscale,
	__half4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx].convert();

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(means[idx], proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (__half2float(proj[0]) * m.x + __half2float(proj[4]) * m.y + __half2float(proj[8]) * m.z + __half2float(proj[12])) * m_w * m_w;
	float mul2 = (__half2float(proj[1]) * m.x + __half2float(proj[5]) * m.y + __half2float(proj[9]) * m.z + __half2float(proj[13])) * m_w * m_w;
	dL_dmean.x = (__half2float(proj[0]) * m_w - __half2float(proj[3]) * mul1) *  __half2float(dL_dmean2D[idx].x) + (__half2float(proj[1]) * m_w - __half2float(proj[3]) * mul2) *  __half2float(dL_dmean2D[idx].y);
	dL_dmean.y = (__half2float(proj[4]) * m_w - __half2float(proj[7]) * mul1) *  __half2float(dL_dmean2D[idx].x) + (__half2float(proj[5]) * m_w - __half2float(proj[7]) * mul2) *  __half2float(dL_dmean2D[idx].y);
	dL_dmean.z = (__half2float(proj[8]) * m_w - __half2float(proj[11]) * mul1) * __half2float(dL_dmean2D[idx].x) + (__half2float(proj[9]) * m_w - __half2float(proj[11]) * mul2) * __half2float(dL_dmean2D[idx].y);

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx].x = __hadd(dL_dmeans[idx].x , __float2half(dL_dmean.x));
	dL_dmeans[idx].y = __hadd(dL_dmeans[idx].y , __float2half(dL_dmean.y));
	dL_dmeans[idx].z = __hadd(dL_dmeans[idx].z , __float2half(dL_dmean.z));

	// Compute gradient updates due to computing colors from SHs
	if (shs){
		glm::vec3 cam_pos_vec3 = { __half2float(campos->x), __half2float(campos->y), __half2float(campos->z) };
		glm::vec3 mean_pos = { __half2float(means[idx].x), __half2float(means[idx].y), __half2float(means[idx].z) };
		glm::vec3 dl_dcolor_vec = { __half2float(dL_dcolor[3 * idx]), __half2float(dL_dcolor[3 * idx + 1]), __half2float(dL_dcolor[3 * idx + 2]) };

		computeColorFromSH(idx, D, M, mean_pos, cam_pos_vec3, (__half3*)shs, clamped, dl_dcolor_vec, dL_dmeans, (__half3*)dL_dsh);
	}

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales){

		glm::vec3 scale = { __half2float(scales[idx].x), __half2float(scales[idx].y), __half2float(scales[idx].z) };
		glm::vec4 rot = { __half2float(rotations[idx].x), __half2float(rotations[idx].y), __half2float(rotations[idx].z), __half2float(rotations[idx].w) };
		computeCov3D(idx, scale, scale_modifier, rot, dL_dcov3D, dL_dscale, dL_drot);
	}
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const __half* __restrict__ bg_color,
	const __half2* __restrict__ points_xy_image,
	const __half4* __restrict__ conic_opacity,
	const __half* __restrict__ colors,
	const __half* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const __half* __restrict__ dL_dpixels,
	__half3* __restrict__ dL_dmean2D,
	__half4* __restrict__ dL_dconic2D,
	__half* __restrict__ dL_dopacity,
	__half* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const __half2 pixf = { __uint2half_rn(pix.x), __uint2half_rn(pix.y) };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ __half2 collected_xy[BLOCK_SIZE];
	__shared__ __half4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ __half collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const __half T_final = inside ? final_Ts[pix_id] : __int2half_rn(0);
	__half T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	__half accum_rec[C] = { __int2half_rn(0) };
	__half dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	__half last_alpha = __int2half_rn(0);
	__half last_color[C] = { __int2half_rn(0) };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const __half ddelx_dx = __float2half(0.5 * W);
	const __half ddely_dy = __float2half(0.5 * H);

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const __half2 xy = collected_xy[j];
			const __half2 d = { __hsub(xy.x , pixf.x), __hsub(xy.y , pixf.y) };
			const __half4 con_o = collected_conic_opacity[j];
			__half power = __hsub(__hmul(__float2half(-0.5f) , __hadd(__hmul(con_o.x , __hmul(d.x , d.x)) , __hmul(con_o.z , __hmul(d.y , d.y)))) , __hmul(con_o.y , __hmul(d.x , d.y)));
			if (__hgt(power , __float2half(0.0f)))
				continue;

			const __half G = hexp(power);
			__half alpha = __hmin(__float2half(0.99f), __hmul(con_o.w , G));
			if (__hlt(alpha , __float2half(1.0f / 255.0f)))
				continue;

			T = __hdiv(T , __hsub(__int2half_rn(1) , alpha));;
			const __half dchannel_dcolor = __hmul(alpha , T);

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			__half dL_dalpha = __float2half(0.0f);
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const __half c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = __hadd(__hmul(last_alpha , last_color[ch]) , __hmul((__hsub(__int2half_rn(1) , last_alpha)) , accum_rec[ch]));
				last_color[ch] = c;

				const __half dL_dchannel = dL_dpixel[ch];
				dL_dalpha = __hadd(dL_dalpha , __hmul(__hsub(c , accum_rec[ch]) , dL_dchannel));
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), __hmul(dchannel_dcolor , dL_dchannel));
			}
			dL_dalpha = __hmul(dL_dalpha,T);
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			__half bg_dot_dpixel = __int2half_rn(0);
			for (int i = 0; i < C; i++)
				bg_dot_dpixel = __hadd(bg_dot_dpixel , __hmul(bg_color[i] , dL_dpixel[i]));
			dL_dalpha = __hadd(dL_dalpha , __hmul(__hdiv(__hneg(T_final) , __hsub(__int2half_rn(1) , alpha)) , bg_dot_dpixel));


			// Helpful reusable temporary variables
			const __half dL_dG = __hmul(con_o.w , dL_dalpha);
			const __half gdx = __hmul(G , d.x);
			const __half gdy = __hmul(G , d.y);
			const __half dG_ddelx = __hsub(__hmul(__hneg(gdx) , con_o.x) , __hmul(gdy , con_o.y));
			const __half dG_ddely = __hsub(__hmul(__hneg(gdy) , con_o.z) , __hmul(gdx , con_o.y));

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, __hmul(dL_dG , __hmul(dG_ddelx , ddelx_dx)));
			atomicAdd(&dL_dmean2D[global_id].y, __hmul(dL_dG , __hmul(dG_ddely , ddely_dy)));

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, __hmul(__hmul(__float2half(-0.5f) , gdx) , __hmul(d.x , dL_dG)));
			atomicAdd(&dL_dconic2D[global_id].y, __hmul(__hmul(__float2half(-0.5f) , gdx) , __hmul(d.y , dL_dG)));
			atomicAdd(&dL_dconic2D[global_id].w, __hmul(__hmul(__float2half(-0.5f) , gdy) , __hmul(d.y , dL_dG)));

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), __hmul(G , dL_dalpha));
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const __half3* means3D,
	const int* radii,
	const __half* shs,
	const bool* clamped,
	const __half3* scales,
	const __half4* rotations,
	const float scale_modifier,
	const __half* cov3Ds,
	const __half* viewmatrix,
	const __half* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const __half3* campos,
	const __half3* dL_dmean2D,
	const __half* dL_dconic,
	__half3* dL_dmean3D,
	__half* dL_dcolor,
	__half* dL_dcov3D,
	__half* dL_dsh,
	__half3* dL_dscale,
	__half4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		radii,
		shs,
		clamped,
		scales,
		rotations,
		scale_modifier,
		projmatrix,
		campos,
		dL_dmean2D,
		dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const __half* bg_color,
	const __half2* means2D,
	const __half4* conic_opacity,
	const __half* colors,
	const __half* final_Ts,
	const uint32_t* n_contrib,
	const __half* dL_dpixels,
	__half3* dL_dmean2D,
	__half4* dL_dconic2D,
	__half* dL_dopacity,
	__half* dL_dcolors)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors
		);
}