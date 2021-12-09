#include "common.h"


//initialize dst with val
//for float
void cu_initializeMemory(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, float val, float* dst, size_t pitch);
//for int
void cu_initializeMemory(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, int val, int* dst, size_t pitch);

//Calculate e^{\alpha}_{targetIndex} (divergence) in Eq. (10)
void cu_calculateDivergence(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, cudaTextureObject_t texture, float* device, size_t pitch);
//Winner Takes All
void cu_updateMinCostAndIndex(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, float* minCost, int* minIndex, float* costSlice, int currentIndex, size_t pitchF, size_t pitchI);
//Estimation confidence calculation
void cu_calculateEstimationConfidence(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, float* minCost, float* sum, float* confidence, float disparityResolution, size_t pitch);

//Apply box filtering to src (window size is windowRadius * 2 + 1)
void cu_boxFiltering(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, int windowRadius, float* src, float* dst, size_t pitch);
//calculate dst += src
void cu_adder(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, float* src, float* dst, size_t pitch);
//Multiply src by val and save it in dst
void cu_scale(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, int* src, int* dst, int val, size_t pitch);

//Find interpolated \bar{\alpha} in Eq. (16)
void cu_findInterpolatedMinAlpha(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, int samplingRate, int currentIndex, int* minIndex, int* interpolatedIndex, float* c_1, float* c0, float* c1, size_t pitchF, size_t pitchI);
//Calculate f^{\alpha}_{viewIndex} and add it to C^{\alpha}_{viewIndex} (= calculate F^{\alpha} (1))
void cu_calculateFeatureWithRemapAndAddToDevice(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, cudaTextureObject_t texture, float* device, size_t pitch, float dx, float dy);
//Calculate h^{\alpha} in Eq. (12)
void cu_calculateCost(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, int numOfViews, float* device, size_t pitch);
//Initialize image
void cu_initializeImage(dim3 blockSize, dim3 gridSize, cudaStream_t stream, int width, int height, float4* src, size_t stepSrc, float4* dst, size_t stepDst);



