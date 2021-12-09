#include "kernel.cuh"

//Image initialization
__global__ void
d_initializeImage(int width, int height, float4* src, size_t stepSrc, float4* dst, size_t stepDst)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idSrc = y * stepSrc + x;
	int idDst = y * stepDst + x;

	if (x >= width || y >= height)
	{
		return;
	}
	//0-1 GBR
	dst[idDst].x = src[idSrc].x / 255.0f;
	dst[idDst].y = src[idSrc].y / 255.0f;
	dst[idDst].z = src[idSrc].z / 255.0f;
	dst[idDst].w = 0.0;
}
void cu_initializeImage(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, float4* src, size_t stepSrc, float4* dst, size_t stepDst)
{
	d_initializeImage << <gridSize, blockSize, 0, stream >> > (width, height, src, stepSrc, dst, stepDst);
}

//Calculate f^{\alpha}_{viewIndex} and add it to C^{\alpha}_{viewIndex} (= calculate F^{\alpha} (1))
__global__ void
de_calculateFeatureWithRemapAndAddToDevice(int width, int height, cudaTextureObject_t texture, float* device, size_t pitch, float pu, float pv)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	//(x,y) is a pixel location on targetView
	//(Xs, Ys) = (pu+x, pv+y)
	float xs = x + pu;
	float ys = y + pv;
	//Feature value of target pixel
	float targetVal = tex2D<float>(texture, xs, ys);

	if(targetVal >= 0)
		*((float*)((char*)device + y * pitch) + x) += 1.0f;
}
void cu_calculateFeatureWithRemapAndAddToDevice(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, cudaTextureObject_t texture, float* device, size_t pitch, float dx, float dy)
{
	de_calculateFeatureWithRemapAndAddToDevice << <gridSize, blockSize, 0, stream >> > (width, height, texture, device, pitch, dx + 0.5f, dy + 0.5f);//add 0.5 for texture sampling
}

//Calculate e^{\alpha}_{targetIndex} (divergence) in Eq. (10)
__global__ void
de_calculateDivergence(int width, int height, cudaTextureObject_t texture, float* device, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
	{
		return;
	}
	//Calculate divergence
	//\nabla x [Is,i] = Is,(x,y) - Is,(x+1,y)
	//\nabla y [Is,i] = Is,(x,y) - Is,(x,y+1)
	//\nabla x [Is,i] + \nabla y [Is,i] = 2 * Is,(x,y) - Is,(x+1,y) - Is,(x,y+1)
	//add 0.5 for texture sampling
	float targetVal = tex2D<float>(texture, x + 0.5f, y + 0.5f) * 2 - tex2D<float>(texture, x + 1.5f, y + 0.5f) -tex2D<float>(texture, x + 0.5f, y + 1.5f);
	*((float*)((char*)device + y * pitch) + x) = targetVal;
}
void cu_calculateDivergence(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, cudaTextureObject_t texture, float* device, size_t pitch)
{
	de_calculateDivergence << <gridSize, blockSize, 0, stream >> > (width, height, texture, device, pitch);
}

//Calculate h^{\alpha} in Eq. (12)
__global__ void
de_calculateCost(int width, int height, int usedViewNum, float* device, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float val = *((float*)((char*)device + y * pitch) + x);
	//F(1) = val
	//F(0) = usedViewNum - val
	//h = F(0) * F(1)
	*((float*)((char*)device + y * pitch) + x) = (usedViewNum - val) * val;
}
void cu_calculateCost(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, int usedViewNum, float* device, size_t pitch)
{
	de_calculateCost << <gridSize, blockSize, 0, stream >> > (width, height, usedViewNum, device, pitch);
}


//Box filtering
__global__ void
de_boxFiltering(int width, int height, int windowRadius, float* src, float* dst, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < 0 || x >= width || y < 0 || y >= height)
	{
		return;
	}

	int pixelNum = 0;
	float sum = 0.0f;

	for (int dy = -windowRadius; dy <= windowRadius; dy++)
	{
		int yy = y + dy;
		if (yy < 0 || yy >= height)
		{
			continue;
		}
		for (int dx = -windowRadius; dx <= windowRadius; dx++)
		{
			int xx = x + dx;
			if (xx < 0 || xx >= width)
			{
				continue;
			}
			sum += *((float*)((char*)src + yy * pitch) + xx);
			pixelNum++;
		}
	}
	*((float*)((char*)dst + y * pitch) + x) = sum / (float)pixelNum;
}
void cu_boxFiltering(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, int windowRadius, float* src, float* dst, size_t pitch)
{
	de_boxFiltering << <gridSize, blockSize, 0, stream >> > (width, height, windowRadius, src, dst, pitch);
}


//Initialize with a value
__global__ void
de_initializeMemory(int width, int height, float val, float* dst, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	*((float*)((char*)dst + y * pitch) + x) = val;
}
void cu_initializeMemory(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, float val, float* dst, size_t pitch)
{
	de_initializeMemory << <gridSize, blockSize, 0, stream >> > (width, height, val, dst, pitch);
}
__global__ void
de_initializeMemory(int width, int height, int val, int* dst, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	*((int*)((char*)dst + y * pitch) + x) = val;
}
void cu_initializeMemory(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, int val, int* dst, size_t pitch)
{
	de_initializeMemory << <gridSize, blockSize, 0, stream >> > (width, height, val, dst, pitch);
}


//calculate dst += src
__global__ void
de_adder(int width, int height, float* src, float* dst, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	*((float*)((char*)dst + y * pitch) + x) += *((float*)((char*)src + y * pitch) + x);
}
void cu_adder(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, float* src, float* dst, size_t pitch)
{
	de_adder << <gridSize, blockSize, 0, stream >> > (width, height, src, dst, pitch);
}

//Multiply src by val and save it in dst
__global__ void
de_scale(int width, int height, int* src, int* dst, int val, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	*((int*)((char*)dst + y * pitch) + x) = *((int*)((char*)src + y * pitch) + x) * val;
}
void cu_scale(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, int* src, int* dst, int val, size_t pitch)
{
	de_scale << <gridSize, blockSize, 0, stream >> > (width, height, src, dst, val, pitch);
}


//Part of process of Wiener Takes All
__global__ void
de_updateMinCostAndIndex(int width, int height, float* minCost, int* minIndex, float* costSlice, int currentIndex, size_t pitchF, size_t pitchI)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float currentVal = *((float*)((char*)costSlice + y * pitchF) + x);
	float minVal = *((float*)((char*)minCost + y * pitchF) + x);
	if (currentVal < minVal) {
		//Update min value
		*((float*)((char*)minCost + y * pitchF) + x) = currentVal;
		//Update min index
		*((int*)((char*)minIndex + y * pitchI) + x) = currentIndex;
	}
}
void cu_updateMinCostAndIndex(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, float* minCost, int* minIndex, float* costSlice, int currentIndex, size_t pitchF, size_t pitchI)
{
	de_updateMinCostAndIndex << <gridSize, blockSize, 0, stream >> > (width, height, minCost, minIndex, costSlice, currentIndex, pitchF, pitchI);
}

//Calculate confidence
__global__ void
de_calculateEstimationConfidence(int width, int height, float* minCost, float* sum, float* confidence, float disparityResolution, size_t pitch)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	float _sum = *((float*)((char*)sum + y * pitch) + x);
	if (_sum <= 0)
		*((float*)((char*)confidence + y * pitch) + x) = 0;//For extreme case
	else
		*((float*)((char*)confidence + y * pitch) + x) = 1 - *((float*)((char*)minCost + y * pitch) + x) * disparityResolution / _sum;
	//*((float*)((char*)confidence + y * pitch) + x) = 1 - *((float*)((char*)minCost + y * pitch) + x) * disparityResolution / *((float*)((char*)sum + y * pitch) + x);

}
void cu_calculateEstimationConfidence(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, float* minCost, float* sum, float* confidence, float disparityResolution, size_t pitch)
{
	de_calculateEstimationConfidence << <gridSize, blockSize, 0, stream >> > (width, height, minCost, sum, confidence, disparityResolution, pitch);
}


//Find interpolated \bar{\alpha} in Eq. (16)
__global__ void
de_findInterpolatedMinAlpha(int width, int height, int samplingRate, int currentIndex, int* minIndex, int* interpolatedIndex, float* c_1, float* c0, float* c1, size_t pitchF, size_t pitchI)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < 0 || x >= width || y < 0 || y >= height)
		return;

	int minIndex_i = *((int*)((char*)minIndex + y * pitchI) + x);
	if (currentIndex == minIndex_i) {
		float delta;
		float c_1_i = *((float*)((char*)c_1 + y * pitchF) + x);
		float c0_i = *((float*)((char*)c0 + y * pitchF) + x);
		float c1_i = *((float*)((char*)c1 + y * pitchF) + x);
		if (c1_i < c_1_i) {
			delta = (c1_i - c_1_i) / (c0_i - c_1_i) / 2;
		}
		else {
			delta = (c1_i - c_1_i) / (c0_i - c1_i) / 2;
		}
		*((int*)((char*)interpolatedIndex + y * pitchI) + x) = minIndex_i * samplingRate + __float2int_rn(delta * (float)samplingRate);
	}
}
void cu_findInterpolatedMinAlpha(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, int samplingRate, int currentIndex, int* minIndex, int* interpolatedIndex, float* c_1, float* c0, float* c1, size_t pitchF, size_t pitchI)
{
	de_findInterpolatedMinAlpha << <gridSize, blockSize, 0, stream >> > (width, height, samplingRate, currentIndex, minIndex, interpolatedIndex, c_1, c0, c1, pitchF, pitchI);
}
