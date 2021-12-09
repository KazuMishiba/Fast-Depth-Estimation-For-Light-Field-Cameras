#pragma once
#include "common.h"


//Settings for image data
class LF_setting
{
public:
	LF_setting::LF_setting(boost::property_tree::ptree pt);
	LF_setting();
	~LF_setting();

	// Sence info from config file (parameters.cfg)
	SceneInfo sceneInfo;

	// All image names
	std::vector<std::string> allImageNames;

	// Current target index
	int currentTargetIndex;
	// Estimate disparity for all viewpoints
	bool estimateAll;
	// Target index specified in json
	int targetIndex;

	//
	void saveFinalResult(cv::Mat alpha, cv::Mat disparity, float runtime);
	void saveInitialResult(cv::Mat alpha, cv::Mat disparity, float runtime);
	void saveResult(cv::Mat alpha, cv::Mat disparity, float runtime);
	void saveResultDisparity(cv::Mat alpha, cv::Mat disparity, std::string fileName);
	void saveResultRuntime(float runtime, std::string fileName);

private:
	// Image dir
	std::string imageDir;
	// Output dir
	std::string outputDir;
	// Name for result
	std::string outputBaseNameEstimation;
	// Name for runtime
	std::string outputBaseNameRuntime;
	// Add viewpoint index to save name
	bool outputUseViewpointIndexAsSuffix;
	// Suffix for initial estimation
	std::string outputSuffixInitialEstimation;
	// Config file of a scene
	std::string configFileName;

	void loadConfig();

	//Save
	bool saveAsPFM;
	bool saveAsPNG;
	bool saveResultInitialEstimation;
	bool saveResultInitialRuntime;
	bool saveResultFinalEstimation;
	bool saveResultFinalRuntime;

	void checkAndCreateDirectory(std::string fileName);
};

