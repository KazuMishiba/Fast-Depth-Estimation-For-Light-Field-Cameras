#pragma once
#include "Parameters.h"
#include "Settings.h"
#include "LF_setting.h"
#include "common.h"
#include <curand_kernel.h>

//Timer
class Timer
{
public:
	Timer();
	~Timer();
	void Timer::start();
	void Timer::stop();
	void Timer::display(std::string title);
	float Timer::getTime();

private:
	cudaEvent_t start_time, stop_time;
	float milliseconds;
};



//Disparity settings
class Disparity {
public:
	float disp_min;
	float disp_max;
	float disp_range;
	int disp_num_withoutSampling;
	int disp_num_withSampling;
	float disp_step_withoutSampling;
	Disparity(float _disp_min, float _disp_max, int disparityResolutionWithoutSampling, int samplingParameter);
	std::vector<float> disparityCandidateWithSampling;
	std::vector<float> disparityCandidateWithoutSampling;
	//
	cv::Mat getDisparityFromParameter(cv::Mat alpha);
	//
	cv::Mat normalizedParameterForDisplay(cv::Mat alpha);
};


class Prop
{
public:
	Prop(Settings& settings);
	~Prop();
	void perform(int targetIndex, bool first = true);
	void setup();
	void setupForInitialEstimation();
	void initialEstimation();
	void setupForOptimization();
	void optimization();


	void readImage(cv::Mat &img, std::string fileName);
	void readViewImage(cv::Mat &img, int n);


	int centerIndex() { return (num - 1) / 2; };
	int numOfAllView() { return num; };

private:
	Settings settings;

	int vn;
	int un;
	int num;
	int width;
	int height;

	//index converter
	//(u,v) = n
	//(0,0) = 0, (1,0) = 1, ..., (un-1,0) = un-1, (0,1) = un, (1,1) = un+1,...
	int sub2ind(int u, int v) { return v * un + u; };
	int2 ind2sub(int n) { return make_int2(n%un, n / un); };

	Disparity *disparityInfo;

	//For CUDA
	dim3 blockSize;
	dim3 gridSize;
	std::vector<cudaStream_t> streams;
	//Pitch of linear memory per image
	size_t pitchF1;//float
	size_t pitchI1;//int


	//Loaded image (normalized to 0~1 as float and grayed (CV_32FC1))
	std::vector<cv::Mat> view_rawHost;
	//Difference (divergence) data on device (float*)
	std::vector<float*> view_divergenceDevice;
	//Difference (divergence) data on host (CV_32FC(4))
	std::vector<cv::Mat> view_divergenceHost;
	//Difference (divergence) data on texture
	std::vector<cudaTextureObject_t> view_divergenceTexture;
	//Estimated disparity index (all viewpoints, after interpolation) on device (int*)
	std::vector<int*> view_disparityIndexDevice;
	//Estimation confidence on device (float*)
	std::vector<float*> view_confidenceDevice;
	//Feature value of target view on device
	float* target_featureDevice;
	//For optimization process
	cv::Mat alpha_initial;//initial result alpha (CV_32SC1)
	cv::Mat guideImage;//Image for weight caluculation in Eq. (9) (CV_32SC1)
	cv::Mat confidenceMap;//Estimation confidence c_i (CV_32FC1)
	//Final disparity estimation result
	cv::Mat alpha_final;//final result alpha (CV_32SC1)
	//Normalized for display
	cv::Mat alpha_initial_normalized;
	cv::Mat alpha_final_normalized;


	//View
	//Relative viewpoint coordinates of each viewpoint from the estimated target viewpoint (x,y)
	std::vector<int2> relativeCoordinatesFromTarget;
	//Viewpoint index (in order of processing, with the first viewpoint being the target of disparity estimation)
	std::vector<int> usedViewIndices;
	//Viewpoint selecion
	void Prop::calculateUsedViewIndices();
	

	//Cost volume on device
	std::vector<float*> costVolume;
	
	//
	void Prop::setRawImage(cv::Mat mat, int viewIndex);
	void Prop::calculateDivergenceAndSetToTexture(int viewIndex);
	void Prop::initializeCostVolume(float* defaultValues);
	void Prop::calculateFeature(int viewIndex, float* featureDevice);
	void Prop::calculateFeatureWithRemapAndAddToCostSlice(int viewIndex, int alpha);
	void Prop::calculateCost(int alpha, int usedViewNum);
	void Prop::aggregateCost(int alpha);
	void Prop::winnerTakesAll(float* minCost, int* minAlpha, int disparityResolution);
	void Prop::calculateEstimationConfidence(float* minCost, float* confidence);
	void Prop::convertIndexFromSampledResolutionToOriginalResolution(int* sampled, int* dst);
	void Prop::findInterpolatedMinAlpha(int* minIndex, int* interpolatedIndex);

	//
	void Prop::freeCudaMemory();


	//Set linear memory to texture
	void Prop::setLinearArrayToTexture(float* dLF1, cudaTextureObject_t& texObj);

	//Allocate memory on the device of the same size as the image size
	void Prop::allocateDeviceMemory(float** dst);
	void Prop::allocateDeviceMemory(int** dst);
	void Prop::allocateDeviceMemory(float** dst, float initialValue, int streamIndex = 0);
	void Prop::allocateDeviceMemory(int** dst, int initialValue, int streamIndex = 0);


	//Display target viewpoint info
	void Prop::showTargetViewpointInfo();

	//Timer
	Timer timer_initial, timer_optimization;

	//Save results
	void Prop::saveResult();

	//For debug
	void Prop::showDevice(float* src, std::string title, bool useAbs, float scalingFactor);
	void Prop::showDevice(int* src, std::string title, bool useAbs, float scalingFactor);

};

