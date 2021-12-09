#include "Prop.h"
#include "kernel.cuh"
#include "l1solver.h"


////////////////////////////////////////
// Utility
////////////////////////////////////////
//GPU memory info
void cu_memoryInfo()
{
	size_t mf, ma;
	cudaMemGetInfo(&mf, &ma);
	cudaDeviceSynchronize();
	std::cout << "[GPU Memory] free: " << mf << " total: " << ma << std::endl << std::endl;
}

//GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		cu_memoryInfo();
		if (abort) exit(code);
	}
}

//Timer
Timer::Timer()
{
	this->milliseconds = 0;
	cudaEventCreate(&start_time);
	cudaEventCreate(&stop_time);
}

Timer::~Timer()
{
	cudaEventDestroy(start_time);
	cudaEventDestroy(stop_time);
}

void Timer::start()
{
	//Record start time
	cudaEventRecord(start_time);
}

void Timer::stop()
{
	//Calculate runtime
	cudaEventRecord(stop_time);
	cudaEventSynchronize(stop_time);
	cudaEventElapsedTime(&milliseconds, start_time, stop_time);
}

void Timer::display(std::string title)
{
	std::cout << title << milliseconds << " [ms]" << std::endl;
}

float Timer::getTime()
{
	return this->milliseconds;
}



////////////////////////////////////////
// Disparity
////////////////////////////////////////
Disparity::Disparity(float _disp_min, float _disp_max, int disparityResolutionWithoutSampling, int samplingParameter) {
	this->disp_min = _disp_min;
	this->disp_max = _disp_max;
	this->disp_range = _disp_max - _disp_min;
	//No sampling
	this->disp_step_withoutSampling = this->disp_range / (float)(disparityResolutionWithoutSampling - 1);
	this->disparityCandidateWithoutSampling.resize(disparityResolutionWithoutSampling);
	for (int i = 0; i < disparityResolutionWithoutSampling; i++)
	{
		this->disparityCandidateWithoutSampling[i] = this->disp_min + this->disp_step_withoutSampling * (float)i;
	}

	//Use cost volume interpolation
	this->disparityCandidateWithSampling.resize(0);
	for (int i = 0; i < disparityResolutionWithoutSampling; i = i + samplingParameter)
	{
		this->disparityCandidateWithSampling.push_back(this->disp_min + this->disp_step_withoutSampling * (float)i);
	}
	this->disp_num_withoutSampling = disparityResolutionWithoutSampling;
	this->disp_num_withSampling = this->disparityCandidateWithSampling.size();
	if (samplingParameter == 1)
	{
		std::cout << "Cost volume sampling: No" << std::endl;
		std::cout << "Disparity resolution: " << disparityResolutionWithoutSampling <<  std::endl;
	}
	else
	{
		std::cout << "Cost volume sampling: Yes" << std::endl;
		std::cout << "Full disparity resolution: " << disparityResolutionWithoutSampling << std::endl;
		std::cout << "Sampled disparity resolution: " << this->disp_num_withSampling << std::endl;
		std::cout << "Sampled index: ";
		for (int i = 0; i < disparityResolutionWithoutSampling; i = i + samplingParameter)
		{
			std::cout << i << " ";
		}
		std::cout << std::endl;

	}
	std::cout << std::endl;
}
cv::Mat Disparity::normalizedParameterForDisplay(cv::Mat alpha)
{
	cv::Mat normalizedAlpha;
	alpha.convertTo(normalizedAlpha, CV_8UC1, 255.0 / (disp_num_withoutSampling-1));
	return normalizedAlpha;
}
cv::Mat Disparity::getDisparityFromParameter(cv::Mat alpha) {
	cv::Mat disparity;
	alpha.convertTo(disparity, CV_32FC1, this->disp_step_withoutSampling, this->disp_min);
	return disparity;
}

////////////////////////////////////////
// Prop
////////////////////////////////////////

//Viewpoint selecion
void Prop::calculateUsedViewIndices() {
	int targetIndex = this->settings.lf_settings.currentTargetIndex;
	if (this->settings.param.viewSelection == "all") {
		std::cout << "viewSelection is \"all\". So usedViewNum is set to " << this->num << "." << std::endl;
		this->settings.param.usedViewNum = this->num;
	}
	if (this->settings.param.usedViewNum > this->num)
	{
		std::cout << "usedViewNum is over the number of all views. So usedViewNum is set to " << this->num << "." << std::endl;
		this->settings.param.usedViewNum = this->num;
	}
	std::cout << std::endl;

	//Absolute coordinates of all viewpoints (top-left viewpoint (0, 0))
	std::vector<int2> absoluteCoordinate(this->num);
	for (int i = 0; i < this->num; i++)
	{
		absoluteCoordinate[i] = ind2sub(i);
	}

	//coordinate of target viewpoint
	int2 targetCoordinate = ind2sub(targetIndex);
	//relative coordinates of each viewpoint from target viewpoint
	this->relativeCoordinatesFromTarget.resize(this->num);
	for (int i = 0; i < this->num; i++)
	{
		this->relativeCoordinatesFromTarget[i] = make_int2((float)(absoluteCoordinate[i].x - targetCoordinate.x), (float)(absoluteCoordinate[i].y - targetCoordinate.y));
		//std::cout << i << " = (" << this->relativeCoordinatesFromTarget[i].x << ", " << this->relativeCoordinatesFromTarget[i].y << ")" << std::endl;
	}

	this->usedViewIndices.resize(this->settings.param.usedViewNum);
	this->usedViewIndices[0] = targetIndex;

	//selected flag
	std::vector<bool> isSelected(this->num, false);
	isSelected[targetIndex] = true;

	if (this->settings.param.viewSelection == "all") {
		//Use all viewpoints
		for (int i = 0, n = 1; i < this->num; i++)
		{
			if (!isSelected[i]) {
				this->usedViewIndices[n] = i;
				isSelected[i] = true;
				n++;
			}
		}
	} else if(this->settings.param.viewSelection == "adaptive")	{
		//Adaptive viewpoint selection (proposed method)

		int vn = this->vn;
		int un = this->un;
		auto checkCase = [](int2 relativeCoordinate) -> bool {
			return relativeCoordinate.x != 0 && relativeCoordinate.y != 0;
		};
		auto checkRange = [=](int i, int j) -> bool {
			int ii = i + targetCoordinate.x;
			int jj = j + targetCoordinate.y;
			return (ii >= 0) && (ii < un) && (jj >= 0) && (jj < vn) ? true : false;
		};
		auto getIndexOfRelativeCoordinate = [=](int i, int j) -> int {
			int ii = i + targetCoordinate.x;
			int jj = j + targetCoordinate.y;
			return (ii >= 0) && (ii < un) && (jj >= 0) && (jj < vn) ? sub2ind(ii, jj) : -1;
		};

		int selectedViewNum = 1;
		while(selectedViewNum < this->settings.param.usedViewNum)
		{
			float minVq = LARGE_VALUE;
			std::vector<int> selectedQs;
			for (int n = 0; n < this->num; n++)
			{
				if (!isSelected[n]) {
					std::vector<int> qs;
					int i = this->relativeCoordinatesFromTarget[n].x;
					int j = this->relativeCoordinatesFromTarget[n].y;
					if (checkCase(this->relativeCoordinatesFromTarget[n])) {
						std::vector<int2> possibleCoordinates{
							make_int2(i,j),
							make_int2(-i,-j),
							make_int2(i,-j),
							make_int2(-i,j) };
						for (auto coord = possibleCoordinates.begin(); coord != possibleCoordinates.end(); ++coord) {
							int index = getIndexOfRelativeCoordinate((*coord).x, (*coord).y);
							if(index != -1 && !isSelected[index])
								qs.push_back(index);
						}

					}
					else {
						int _i = i != 0 ? i : j;
						std::vector<int2> possibleCoordinates{
							make_int2(_i,0),
							make_int2(-_i,0),
							make_int2(0,_i),
							make_int2(0, -_i) };
						for (auto coord = possibleCoordinates.begin(); coord != possibleCoordinates.end(); ++coord) {
							int index = getIndexOfRelativeCoordinate((*coord).x, (*coord).y);
							if (index != -1 && !isSelected[index])
								qs.push_back(index);
						}
					}
					//Calculate V(q)
					float vq = 0.0f;
					for (auto s = qs.begin(); s != qs.end(); ++s) {
						int2 sCoord = this->relativeCoordinatesFromTarget[*s];
						//first term
						vq += this->settings.param.gamma * (float)(std::abs(sCoord.x) + std::abs(sCoord.y));

						int sumDif = 0;
						for (int nn = 0; nn < this->num; nn++)
						{
							if (isSelected[nn]) {
								//second term
								int difX = std::abs(absoluteCoordinate[n].x - absoluteCoordinate[nn].x);
								int difY = std::abs(absoluteCoordinate[n].y - absoluteCoordinate[nn].y);
								sumDif += difX + difY;
							}
						}
						vq -= (float)sumDif / (float)selectedViewNum;
					}
					//use average instead of sum
					//The reason for using average instead of sum is to make it work better when the target viewpoint for estimation is not the central viewpoint.
					//(The paper discusses disparity estimation for the central viewpoint only.)
					vq /= (float)qs.size();

					//update
					if (vq < minVq) {
						minVq = vq;
						selectedQs.resize(qs.size());
						std::copy(qs.begin(), qs.end(), selectedQs.begin());
					}
				}
			}
			//Add viewpoints into index list
			for (auto s = selectedQs.begin(); s != selectedQs.end(); ++s) {
				this->usedViewIndices[selectedViewNum] = *s;
				isSelected[*s] = true;
				selectedViewNum++;
				if (selectedViewNum >= this->settings.param.usedViewNum)
					break;
			}
		}
	} else {
		std::cerr << "viewSelection is not set correctly. \n";
		throw;
	}

	//Display
	std::cout << "Used View Indices: ";
	for (auto &v: this->usedViewIndices)
	{
		std::cout << v << " ";
	}
	std::cout << std::endl;
	if (this->settings.reduceData.enable && this->settings.reduceData.enableLimitView)
	{
		std::cout << "(The index here is the number of viewpoint, with 0 being the upper left corner of the image set to be used. " <<
			"If ReduceData.enable == 1 and ReduceData.enableLimitiView == 1, this number will not match the number of the image to be loaded.)"
			<< std::endl;
	}
}



void Prop::perform(int targetIndex, bool first)
{
	this->settings.lf_settings.currentTargetIndex = targetIndex;
	//Target viewpoint info
	this->showTargetViewpointInfo();

	//Calculate viewpoints used
	this->calculateUsedViewIndices();


	//Initial estimation
	std::cout << "=== Start initial estimation ===" << std::endl;
	this->timer_initial.start();
	if(first)
		this->setupForInitialEstimation();
	this->initialEstimation();
	this->timer_initial.stop();
	this->timer_initial.display("Initial estimation time:");
	std::cout << "=== End initial estimation ===" << std::endl;

	
	//Optimization
	if (this->settings.param.useOptimization)
	{
		std::cout << "=== Start optimization ===" << std::endl;
		this->timer_optimization.start();
		this->setupForOptimization();
		this->optimization();
		this->timer_optimization.stop();
		this->timer_optimization.display("Optimization time:");
		std::cout << "=== End optimization ===" << std::endl;
	}
	else {
		//Set the initial value (initial estimation result) into alpha_initial as CV_32SC1.
		this->alpha_initial = cv::Mat(this->height, this->width, CV_32SC1);
		cudaMemcpy2D(this->alpha_initial.data, this->alpha_initial.step, this->view_disparityIndexDevice[this->settings.lf_settings.currentTargetIndex], this->pitchI1, this->width * sizeof(int), this->height, cudaMemcpyDefault);
		std::cout << "=== Skip optimization ===" << std::endl;
	}

	//Computation time
	std::cout << "Total time:" << this->timer_initial.getTime() + this->timer_optimization.getTime() << " [ms]" << std::endl;

	//Normalize for display
	this->alpha_initial_normalized = this->disparityInfo->normalizedParameterForDisplay(this->alpha_initial);
	if (this->settings.param.useOptimization)
		this->alpha_final_normalized = this->disparityInfo->normalizedParameterForDisplay(this->alpha_final);


	//Save
	this->saveResult();

	//Show
	if (this->settings.displayResult)
	{
		cv::imshow("Initial result", this->alpha_initial_normalized);
		//Show final result
		if (this->settings.param.useOptimization)
			cv::imshow("Final result", this->alpha_final_normalized);
		cv::waitKey(0);
	}
	
	//Free memory for all viewpoint estimation
	this->freeCudaMemory();
}


//Load image
void Prop::readImage(cv::Mat &img, std::string fileName)
{
	img = cv::imread(fileName, cv::IMREAD_COLOR);
	if (img.empty())
	{
		std::cerr << "Load error: " + fileName << std::endl;
		throw;
	}
	//Data reduction for debug
	if (this->settings.reduceData.enableScaleImageSize)
	{
		//Bicubic interpolation
		cv::resize(img, img, cv::Size(), this->settings.reduceData.scalingRate, this->settings.reduceData.scalingRate, cv::INTER_CUBIC);
	}
}

//Load image
void Prop::readViewImage(cv::Mat &img, int n)
{
	this->readImage(img, this->settings.lf_settings.allImageNames[this->settings.currentSceneInfo.imageIndex[n]]);
}

//Convert loaded image CV_8UC3 to CV_32F1 of 0 to 1 and store it to view_rawHost[viewIndex]
void Prop::setRawImage(cv::Mat mat, int viewIndex) {
	cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
	mat.convertTo(this->view_rawHost[viewIndex], CV_32FC1, 1. / 255);
}




Prop::Prop(Settings& settings)
{
	cu_memoryInfo();

	this->settings = settings;

	this->vn = this->settings.currentSceneInfo.vn;
	this->un = this->settings.currentSceneInfo.un;
	this->num = this->settings.currentSceneInfo.num;
	this->height = this->settings.currentSceneInfo.height;
	this->width = this->settings.currentSceneInfo.width;
	

    //Settings for CUDA
	this->blockSize = dim3(4, 4, 1);
	int gridSizeX = ceil(width / (float)blockSize.x);
	int gridSizeY = ceil(height / (float)blockSize.y);
	int gridSizeZ = 1;
	this->gridSize = dim3(gridSizeX, gridSizeY, gridSizeZ);

	//Get pitch info

	//float*
	float* dst_f1;
	gpuErrchk(cudaMallocPitch(&dst_f1, &this->pitchF1, this->width * sizeof(float), this->height));
	cudaFree(dst_f1);
	//int*
	int* dst_i1;
	gpuErrchk(cudaMallocPitch(&dst_i1, &this->pitchI1, this->width * sizeof(int), this->height));
	cudaFree(dst_i1);


	//Setup
	this->setup();

	//Disparity info
	this->disparityInfo = new Disparity(this->settings.currentSceneInfo.disp_min, this->settings.currentSceneInfo.disp_max, this->settings.param.disparityResolution, this->settings.param.t);
}

Prop::~Prop()
{
	//Free memory
	delete this->disparityInfo;

	for (int alpha = 0; alpha < this->settings.param.disparityResolution; alpha++)
	{
		cudaFree(this->costVolume[alpha]);
	}
	for (int n = 0; n < this->num; n++)
	{
		cudaStreamDestroy(this->streams[n]);
	}
}

//Setup
void Prop::setup()
{
	//Data
	this->streams.resize(this->num);
	this->view_rawHost.resize(this->num);
	this->view_divergenceHost.resize(this->num);
	this->view_divergenceDevice.resize(this->num);
	this->view_divergenceTexture.resize(this->num);
	this->view_disparityIndexDevice.resize(this->num);
	this->view_confidenceDevice.resize(this->num);
	for (int n = 0; n < this->num; n++)
	{
			cudaStreamCreate(&this->streams[n]);
			this->view_divergenceTexture[n] = cudaTextureObject_t(0);
	}

	//Load images
	for (int n = 0; n < this->num; n++)
	{
		cv::Mat mat;
		this->readViewImage(mat, n);

		//Save image as Mat in grayscale image to view_rawHost
		this->setRawImage(mat, n);
	}
	
	cudaDeviceSynchronize();
}

////////////////////////////////////////
// Initial estimation
////////////////////////////////////////

//Setup for initial estimation
void Prop::setupForInitialEstimation()
{
	this->costVolume.resize(this->settings.param.disparityResolution);
	//Allocate memory for cost volume
	for (int alpha = 0; alpha < this->settings.param.disparityResolution; alpha++)
	{
		//Allocate linear memory on device as cudaMallocPitch (float) 
		this->allocateDeviceMemory(&this->costVolume[alpha]);
	}
}


//Calculate f^{\alpha}_{s}
void Prop::calculateFeature(int viewIndex, float* featureDevice) {
	//Set the divergence of the viewpoint (viewIndex) to view_divergenceTexture
	this->setLinearArrayToTexture(this->view_divergenceDevice[viewIndex], this->view_divergenceTexture[viewIndex]);

	//Calculate the feature value by sign judgment, and add the result(0 or 1) to view_featureDevice.
	cu_calculateFeatureWithRemapAndAddToDevice(this->blockSize, this->gridSize, this->streams[viewIndex], this->width, this->height, this->view_divergenceTexture[viewIndex], featureDevice, this->pitchF1, 0.0f, 0.0f);
}

//Initialize each slice of the cost volume with defaultValues
void Prop::initializeCostVolume(float* defaultValues) {
	for (int i = 0; i < this->costVolume.size(); i++)
	{
		//device to device
		cudaMemcpy2DAsync(this->costVolume[i], this->pitchF1, defaultValues, this->pitchF1,width * sizeof(float), height, cudaMemcpyDefault, this->streams[i%this->num]);
	}
}

//Calculate f^{\alpha}_{viewIndex} and add it to C^{\alpha}_{viewIndex} (= calculate F^{\alpha} (1))
void Prop::calculateFeatureWithRemapAndAddToCostSlice(int viewIndex, int alpha) {
	//Calculate pu and pv in Eq. (1) for remap
	float pu = - (float)this->relativeCoordinatesFromTarget[viewIndex].x * this->disparityInfo->disparityCandidateWithSampling[alpha];
	float pv = - (float)this->relativeCoordinatesFromTarget[viewIndex].y * this->disparityInfo->disparityCandidateWithSampling[alpha];

	cu_calculateFeatureWithRemapAndAddToDevice(this->blockSize, this->gridSize, this->streams[viewIndex], this->width, this->height, this->view_divergenceTexture[viewIndex], this->costVolume[alpha], this->pitchF1, pu, pv);
}

//Calculate e^{\alpha}_{viewIndex} in Eq. (10)
void Prop::calculateDivergenceAndSetToTexture(int viewIndex) {
	float* view_rawDevice;
	this->allocateDeviceMemory(&view_rawDevice);
	cudaMemcpy2D(view_rawDevice, this->pitchF1, this->view_rawHost[viewIndex].data, this->view_rawHost[viewIndex].step, this->width * sizeof(float), this->height, cudaMemcpyDefault);
	cudaTextureObject_t view_rawTexture;
	this->setLinearArrayToTexture(view_rawDevice, view_rawTexture);

	//Calculate difference (divergence) on texture and record it in this->view_divergenceDevice[viewIndex
	this->allocateDeviceMemory(&this->view_divergenceDevice[viewIndex]);
	cu_calculateDivergence(this->blockSize, this->gridSize, this->streams[viewIndex], this->width, this->height, view_rawTexture, this->view_divergenceDevice[viewIndex], this->pitchF1);

	//Set divergence to texture
	this->setLinearArrayToTexture(this->view_divergenceDevice[viewIndex], this->view_divergenceTexture[viewIndex]);

	cudaFree(view_rawDevice);
	cudaDestroyTextureObject(view_rawTexture);
}


//Calculate h^{\alpha} in Eq. (12)
void Prop::calculateCost(int alpha, int usedViewNum) {
	cu_calculateCost(this->blockSize, this->gridSize, this->streams[alpha%this->num], this->width, this->height, usedViewNum, this->costVolume[alpha], this->pitchF1);
}




//Cost aggregation (box filtering)
void Prop::aggregateCost(int alpha) {
	int windowRadius = (this->settings.param.W1 - 1) / 2;

	//Allocate memory for aggregation results
	float* dst;
	this->allocateDeviceMemory(&dst);

	//Aggregate cost using box filtering
	cu_boxFiltering(this->blockSize, this->gridSize, this->streams[alpha%this->num], this->width, this->height, windowRadius, this->costVolume[alpha], dst, this->pitchF1);

	cudaMemcpy2D(this->costVolume[alpha], this->pitchF1, dst, this->pitchF1, this->width * sizeof(float), this->height, cudaMemcpyDefault);
	cudaFree(dst);
}

//Wiener takes all (Eq. (15))
void Prop::winnerTakesAll(float* minCost, int* minAlpha, int disparityResolution) {
	for (int alpha = 0; alpha < disparityResolution; alpha++)
	{
		cu_updateMinCostAndIndex(this->blockSize, this->gridSize, this->streams[alpha%this->num], this->width, this->height, minCost, minAlpha, this->costVolume[alpha], alpha, this->pitchF1, this->pitchI1);
	}
}

//Calculate estimation confidence
void Prop::calculateEstimationConfidence(float* minCost, float* confidence){

	//Allocate memory for ave C
	float* sum;
	this->allocateDeviceMemory(&sum);
	//Initialize with 0
	cu_initializeMemory(this->blockSize, this->gridSize, this->streams[0], this->width, this->height, 0.0f, sum, this->pitchF1);
	//Calculate sum of C
	for (int alpha = 0; alpha < this->disparityInfo->disp_num_withSampling; alpha++)
	{
		cu_adder(this->blockSize, this->gridSize, this->streams[0], this->width, this->height, this->costVolume[alpha], sum, this->pitchF1);
	}

	//Calculate estimation confidence
	cu_calculateEstimationConfidence(this->blockSize, this->gridSize, this->streams[0], this->width, this->height, minCost, sum, confidence, (float)this->disparityInfo->disp_num_withSampling, this->pitchF1);

	cudaFree(sum);
}

//Calculate \hat{\alpha} in original resolution
void Prop::convertIndexFromSampledResolutionToOriginalResolution(int* sampled, int* dst) {
	cu_scale(this->blockSize, this->gridSize, this->streams[0], this->width, this->height, sampled, dst, this->settings.param.t, this->pitchI1);
}

//Find interpolated \bar{\alpha} in Eq. (16)
void Prop::findInterpolatedMinAlpha(int* minIndex, int* interpolatedIndex) {
	for (int alpha = 1; alpha < this->disparityInfo->disp_num_withSampling - 1; alpha++)
	{
		cu_findInterpolatedMinAlpha(this->blockSize, this->gridSize, this->streams[alpha%this->num], this->width, this->height, this->settings.param.t, alpha, minIndex, interpolatedIndex, this->costVolume[alpha-1], this->costVolume[alpha], this->costVolume[alpha+1], this->pitchF1, this->pitchI1);
	}
}

//Initial estimation
void Prop::initialEstimation()
{	

	//Target viewpoint index for disparity estimation
	int targetIndex = this->usedViewIndices[0];
	std::cout << "Target view index: " << targetIndex << std::endl;
	//Calculate e^{\alpha}_{targetIndex} (divergence) in Eq. (10) for target index
	//Note that e^{\alpha}_{targetIndex} for all \alpha \in A is the same because remap operations for the target index produce the same result.
	this->calculateDivergenceAndSetToTexture(targetIndex);

	//Display e^{\alpha}_{targetIndex} for debug
	this->showDevice(this->view_divergenceDevice[targetIndex], "Target Divergence", true, 25.0f);

	//Calculate f^{\alpha}_{targetIndex}
	this->allocateDeviceMemory(&this->target_featureDevice, 0.0f, targetIndex);//initialize with 0
	this->calculateFeature(targetIndex, this->target_featureDevice);

	//Display f^{\alpha}_{targetIndex} for debug
	this->showDevice(this->target_featureDevice, "Target Feature", false, 1.0f);

	cudaStreamSynchronize(this->streams[targetIndex]);

	std::cout << "Construct cost volume... " << std::endl;
	
	//Initialize cost volume with f^{\alpha}_{targetIndex} (F(1)).
	this->initializeCostVolume(this->target_featureDevice);

	cudaDeviceSynchronize();

	//Iterate through the remaining viewpoints
	for (int s = 1; s < this->settings.param.usedViewNum; s++)
	{
		if (this->settings.displayIntermediate)
		{
			std::cout << s + 1 << "/" << this->settings.param.usedViewNum << std::endl;
		}
		int viewIndex = this->usedViewIndices[s];
		//Calculate e (divergence) in Eq. (10)
		this->calculateDivergenceAndSetToTexture(viewIndex);

		for (int alpha = 0; alpha < this->disparityInfo->disp_num_withSampling; alpha++)
		{
			//Calculate f^{\alpha}_{viewIndex} and add it to C^{\alpha}_{viewIndex} (= calculate F^{\alpha} (1))
			this->calculateFeatureWithRemapAndAddToCostSlice(viewIndex, alpha);
			cudaDeviceSynchronize();
			//this->showFloatDevice(this->costVolume[alpha], "Cost slice: ", false, 1.0f / this->settings.param.usedViewNum * 5.0);
		}
	}
	cudaDeviceSynchronize();

	//Calculate h^{\alpha} in Eq. (12) and aggregate cost.
	for (int alpha = 0; alpha < this->disparityInfo->disp_num_withSampling; alpha++)
	{
		//Calculate h^{\alpha}
		this->calculateCost(alpha, this->settings.param.usedViewNum);

		//Calculate C^{\alpha} with cost aggregation (box filtering)
		this->aggregateCost(alpha);
	}

	//Initialize variables for wiener takes all
	float* minCost;
	int* minAlpha;
	this->allocateDeviceMemory(&minCost, LARGE_VALUE, 0);
	this->allocateDeviceMemory(&minAlpha);
	cudaStreamSynchronize(this->streams[0]);

	//Wiener takes all (Eq. (15))
	this->winnerTakesAll(minCost, minAlpha, this->disparityInfo->disp_num_withSampling);

	//Calculate estimation confidence in Eq. (8)
	this->allocateDeviceMemory(&this->view_confidenceDevice[targetIndex]);
	this->calculateEstimationConfidence(minCost, this->view_confidenceDevice[targetIndex]);

	//Display estimation confidence for debug
	this->showDevice(this->view_confidenceDevice[targetIndex], "Estimation confidence", false, 1.0f);


	//Cost volume interpolation
	//Allocate memory
	this->allocateDeviceMemory(&this->view_disparityIndexDevice[targetIndex]);
	//Calculate \hat{\alpha} in Eq. (15) in original resolution
	this->convertIndexFromSampledResolutionToOriginalResolution(minAlpha, this->view_disparityIndexDevice[targetIndex]);
	//Find interpolated \bar{\alpha} in Eq. (16)
	this->findInterpolatedMinAlpha(minAlpha, this->view_disparityIndexDevice[targetIndex]);

	//Disparity index display after interpolation for debug
	this->showDevice(this->view_disparityIndexDevice[targetIndex], "Interpolated Alpha", false, 1.0f / (float)(this->disparityInfo->disp_num_withoutSampling - 1));


	cudaFree(minCost);
	cudaFree(minAlpha);
	cudaFree(this->target_featureDevice);
}




////////////////////////////////////////
// Optimization
////////////////////////////////////////

//Setup for optimization
void Prop::setupForOptimization()
{
	int targetIndex = this->settings.lf_settings.currentTargetIndex;
	//Set the initial value (initial estimation result) for optimization into alpha_initial as CV_32SC1.
	this->alpha_initial = cv::Mat(this->height, this->width, CV_32SC1);
	cudaMemcpy2D(this->alpha_initial.data, this->alpha_initial.step, this->view_disparityIndexDevice[targetIndex], this->pitchI1, this->width * sizeof(int), this->height, cudaMemcpyDefault);
	//Set the grayscale image of the target image as a guide image as CV_32SC1
	view_rawHost[targetIndex].convertTo(this->guideImage, CV_32SC1, 255.0f);
	//Set the confidence as CV_32FC1
	this->confidenceMap = cv::Mat(this->height, this->width, CV_32FC1);
	cudaMemcpy2D(this->confidenceMap.data, this->confidenceMap.step, this->view_confidenceDevice[targetIndex], this->pitchF1, this->width * sizeof(float), this->height, cudaMemcpyDefault);
}

//Optimization
void Prop::optimization()
{
	int nI = this->disparityInfo->disp_num_withoutSampling;
	int nF = 256;//resolution of guide image for calculating w_{i,j}
	int r = (this->settings.param.W2 - 1) / 2;

	//Multiresolution solver
	this->alpha_final = l1Solver::solverInterfacePropWithMultiresolution(this->alpha_initial, this->guideImage, r, this->settings.param.sigma, nI, nF, this->confidenceMap, this->settings.param.lambda, this->settings.param.mu0_lowres, this->settings.param.mu0_highres, this->settings.param.kappa, this->settings.param.tau);
}

//
void Prop::freeCudaMemory()
{
	int targetIndex = this->settings.lf_settings.currentTargetIndex;
	for (int s = 0; s < this->settings.param.usedViewNum; s++)
	{
		cudaFree(this->view_divergenceDevice[this->usedViewIndices[s]]);
		cudaDestroyTextureObject(this->view_divergenceTexture[s]);
	}
	cudaFree(this->view_confidenceDevice[targetIndex]);
	cudaFree(this->view_disparityIndexDevice[targetIndex]);
}


////////////////////////////////////////
// Other functions
////////////////////////////////////////
//Display target viewpoint info
void Prop::showTargetViewpointInfo()
{
	std::cout << "### Target viewpoint information ###" << std::endl;
	std::cout << "Index: " << this->settings.lf_settings.currentTargetIndex << std::endl;
	if (this->settings.reduceData.enable && this->settings.reduceData.enableLimitView)
	{
		std::cout << "(The index here is the number of the target for disparity estimation, with 0 being the upper left corner of the image set to be used. " <<
			"If ReduceData.enable == 1 and ReduceData.enableLimitiView == 1, this number will not match the number of the image to be loaded.)"
			 << std::endl;
	}
	std::cout << "Image name: " << this->settings.lf_settings.allImageNames[this->settings.currentSceneInfo.imageIndex[this->settings.lf_settings.currentTargetIndex]] << std::endl;
}

//Save
void Prop::saveResult()
{
	this->settings.lf_settings.saveInitialResult(this->alpha_initial_normalized, this->disparityInfo->getDisparityFromParameter(this->alpha_initial), this->timer_initial.getTime());
	if (this->settings.param.useOptimization)
		this->settings.lf_settings.saveFinalResult(this->alpha_final_normalized, this->disparityInfo->getDisparityFromParameter(this->alpha_final), this->timer_initial.getTime()+this->timer_optimization.getTime());
}

//Download data on device to host and display it for debug
void Prop::showDevice(float* src, std::string title, bool useAbs, float scalingFactor) {
	if (this->settings.displayIntermediate)
	{
		cudaDeviceSynchronize();
		cv::Mat mat(this->height, this->width, CV_32FC1);
		cudaMemcpy2D(mat.data, mat.step, src, this->pitchF1, this->width * sizeof(float), this->height, cudaMemcpyDefault);
		cv::Mat matDisp = mat * scalingFactor;
		cv::imshow(title, useAbs ? cv::abs(matDisp) : matDisp);
		cv::waitKey(0);
	}
}
void Prop::showDevice(int* src, std::string title, bool useAbs, float scalingFactor) {
	if (this->settings.displayIntermediate)
	{
		cudaDeviceSynchronize();
		cv::Mat mat(this->height, this->width, CV_32SC1);
		cudaMemcpy2D(mat.data, mat.step, src, this->pitchI1, this->width * sizeof(int), this->height, cudaMemcpyDefault);
		mat.convertTo(mat, CV_32FC1, 1.0f);
		cv::Mat matDisp = mat * scalingFactor;
		cv::imshow(title, useAbs ? cv::abs(matDisp) : matDisp);
		cv::waitKey(0);
	}
}



//Allocate memory on the device of the same size as the image size
//Give initialValue if you want to initialize it
void Prop::allocateDeviceMemory(float** dst) {
	gpuErrchk(cudaMallocPitch(dst, &this->pitchF1, this->width * sizeof(float), this->height));
}
void Prop::allocateDeviceMemory(int** dst) {
	gpuErrchk(cudaMallocPitch(dst, &this->pitchI1, this->width * sizeof(int), this->height));
}
void Prop::allocateDeviceMemory(float** dst, float initialValue, int streamIndex) {
	gpuErrchk(cudaMallocPitch(dst, &this->pitchF1, this->width * sizeof(float), this->height));
	cu_initializeMemory(this->blockSize, this->gridSize, this->streams[streamIndex], this->width, this->height, initialValue, *dst, this->pitchF1);
}
void Prop::allocateDeviceMemory(int** dst, int initialValue, int streamIndex) {
	gpuErrchk(cudaMallocPitch(dst, &this->pitchI1, this->width * sizeof(int), this->height));
	cu_initializeMemory(this->blockSize, this->gridSize, this->streams[streamIndex], this->width, this->height, initialValue, *dst, this->pitchI1);
}

//Set linear memory to texture
void Prop::setLinearArrayToTexture(float* dLF1, cudaTextureObject_t& texObj)
{
	if (dLF1 == nullptr) {
		std::cerr << "Null ptr." << std::endl;
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
	//
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr = dLF1;
	resDesc.res.pitch2D.desc = channelDesc;
	resDesc.res.pitch2D.width = this->width;
	resDesc.res.pitch2D.height = this->height;
	resDesc.res.pitch2D.pitchInBytes = this->pitchF1;
	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	// Create texture object
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
}

