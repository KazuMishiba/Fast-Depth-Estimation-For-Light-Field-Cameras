#pragma once

#include "LF_setting.h"
#include <regex>

LF_setting::LF_setting() {}

LF_setting::LF_setting(boost::property_tree::ptree pt)
{
	//Path
	std::string path1 = pt.get<std::string>("Path.p1");
	std::string path2 = pt.get<std::string>("Path.p2");
	std::string path3 = pt.get<std::string>("Path.p3");

	//Input
	std::string inputDir = pt.get<std::string>("Input.dir");
	std::string inputBaseName = pt.get<std::string>("Input.baseName");
	this->estimateAll = pt.get<int>("Input.estimateAll") == 1;
	this->targetIndex = pt.get<int>("Input.targetIndex");
	std::string configFileName = pt.get<std::string>("Input.configFilePath");

	//Output
	std::string outputDir = pt.get<std::string>("Output.dir");
	std::string outputBaseNameEstimation = pt.get<std::string>("Output.baseNameEstimation");
	std::string outputBaseNameRuntime = pt.get<std::string>("Output.baseNameRuntime");
	this->outputUseViewpointIndexAsSuffix = pt.get<int>("Output.useViewpointIndexAsSuffix") == 1;
	std::string outputSuffixInitialEstimation = pt.get<std::string>("Output.suffixInitialEstimation");
	this->saveResultInitialEstimation = pt.get<int>("Output.saveResultInitialEstimation") == 1;
	this->saveResultInitialRuntime = pt.get<int>("Output.saveResultInitialRuntime") == 1;
	this->saveResultFinalEstimation = pt.get<int>("Output.saveResultFinalEstimation") == 1;
	this->saveResultFinalRuntime = pt.get<int>("Output.saveResultFinalRuntime") == 1;
	this->saveAsPFM = pt.get<int>("Output.saveAsPFM") == 1;
	this->saveAsPNG = pt.get<int>("Output.saveAsPNG") == 1;


	std::vector<std::string*> strs = { &inputDir, &inputBaseName, &configFileName, &outputDir, &outputBaseNameEstimation, &outputBaseNameRuntime, &outputSuffixInitialEstimation };
	//Replace path
	if (path1 != "")
		for (const auto& str : strs)
			*str = std::regex_replace(*str, std::regex("#p1"), path1);
	if (path2 != "")
		for (const auto& str : strs)
			*str = std::regex_replace(*str, std::regex("#p2"), path2);
	if (path3 != "")
		for (const auto& str : strs)
			*str = std::regex_replace(*str, std::regex("#p3"), path3);



	this->imageDir = inputDir;
	this->outputDir = outputDir;
	this->outputBaseNameEstimation = outputBaseNameEstimation;
	this->outputBaseNameRuntime = outputBaseNameRuntime;
	this->outputSuffixInitialEstimation = outputSuffixInitialEstimation;
	this->configFileName = configFileName;

	//Load config file
	this->loadConfig();

	//Set image file names
	this->allImageNames.resize(this->sceneInfo.num);
	for (int n = 0; n < this->sceneInfo.num; n++)
	{
		std::ostringstream ss;
		ss << std::setw(3) << std::setfill('0') << n;
		this->allImageNames[n] = this->imageDir + inputBaseName + ss.str() + ".png";
	}
}

LF_setting::~LF_setting()
{
}

//Create directory for fileName
void LF_setting::checkAndCreateDirectory(std::string fileName)
{
	std::filesystem::path p = std::filesystem::path(fileName);
	p.remove_filename();
	if (!std::filesystem::exists(p))
	{
		if(!std::filesystem::create_directories(p))
			std::cout << "Fail to create output directory: " << p << std::endl;
		else
			std::cout << "Create output directory:" << p << std::endl;
	}
}

void LF_setting::loadConfig()
{
	// Load sence info from config file (parameters.cfg)
	boost::property_tree::ptree pt;
	try
	{
		std::cout << "### Original scene info ###" << std::endl;

		boost::property_tree::read_ini(this->configFileName, pt);

		this->sceneInfo.width = pt.get<int>("intrinsics.image_resolution_x_px");
		std::cout << "image_resolution_x_px : " << this->sceneInfo.width << std::endl;

		this->sceneInfo.height = pt.get<int>("intrinsics.image_resolution_y_px");
		std::cout << "image_resolution_y_px : " << this->sceneInfo.height << std::endl;

		this->sceneInfo.un = pt.get<int>("extrinsics.num_cams_x");
		std::cout << "num_cams_x : " << this->sceneInfo.un << std::endl;

		this->sceneInfo.vn = pt.get<int>("extrinsics.num_cams_y");
		std::cout << "num_cams_y : " << this->sceneInfo.vn << std::endl;

		this->sceneInfo.disp_min = pt.get<float>("meta.disp_min");
		std::cout << "disp_min : " << this->sceneInfo.disp_min << std::endl;

		this->sceneInfo.disp_max = pt.get<float>("meta.disp_max");
		std::cout << "disp_max : " << this->sceneInfo.disp_max << std::endl << std::endl;

		//Total number of viewpoints
		this->sceneInfo.num = this->sceneInfo.vn * this->sceneInfo.un;

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		std::wcout << L"ERROR : " << e.what() << std::endl;
		exit(1);
	}
	catch (boost::property_tree::ini_parser_error& e) {
		std::wcout << L"ERROR : " << e.what() << std::endl;
		exit(1);
	}


}

//Save initial result
void LF_setting::saveInitialResult(cv::Mat alpha, cv::Mat disparity, float runtime)
{
	std::string sfx = "";
	if (this->outputUseViewpointIndexAsSuffix)
	{
		std::ostringstream ss;
		ss << std::setw(3) << std::setfill('0') << this->currentTargetIndex;
		sfx = ss.str();
	}

	if (this->saveResultInitialEstimation)
	{
		std::string fileName = this->outputDir + this->outputBaseNameEstimation + sfx + this->outputSuffixInitialEstimation;
		this->checkAndCreateDirectory(fileName);
		this->saveResultDisparity(alpha, disparity, fileName);
	}
	if (this->saveResultInitialRuntime)
	{
		std::string fileName = this->outputDir + this->outputBaseNameRuntime + sfx + this->outputSuffixInitialEstimation;
		this->checkAndCreateDirectory(fileName);
		this->saveResultRuntime(runtime, fileName);
	}

}
//Save final result
void LF_setting::saveFinalResult(cv::Mat alpha, cv::Mat disparity, float runtime)
{
	std::string sfx = "";
	if (this->outputUseViewpointIndexAsSuffix)
	{
		std::ostringstream ss;
		ss << std::setw(3) << std::setfill('0') << this->currentTargetIndex;
		sfx = ss.str();
	}

	if (this->saveResultFinalEstimation)
	{
		std::string fileName = this->outputDir + this->outputBaseNameEstimation + sfx;
		this->checkAndCreateDirectory(fileName);
		this->saveResultDisparity(alpha, disparity, fileName);
	}
	if (this->saveResultFinalRuntime)
	{
		std::string fileName = this->outputDir + this->outputBaseNameRuntime + sfx;
		this->checkAndCreateDirectory(fileName);
		this->saveResultRuntime(runtime, fileName);
	}
}
//Save disparity
void LF_setting::saveResultDisparity(cv::Mat alpha, cv::Mat disparity, std::string fileName)
{
	//Save in PFM
	if (this->saveAsPFM)
	{
		std::string fn = fileName + ".pfm";
		if (cv::imwrite(fn, disparity)) 
			std::cout << "Saved: " << fn << std::endl;
		else
			std::cout << "Failed to save: " << fn << std::endl;
	}
	//Save in PNG
	if (this->saveAsPNG)
	{
		std::string fn = fileName + ".png";
		if (cv::imwrite(fn, alpha))
			std::cout << "Saved: " << fn << std::endl;
		else
			std::cout << "Failed to save: " << fn << std::endl;

	}
}

//Save runtime in sec
void LF_setting::saveResultRuntime(float runtime, std::string fileName)
{
	std::string fn = fileName + ".txt";
	std::ofstream outputfile(fn);
	if (outputfile)
	{
		outputfile << runtime / 1000;
		outputfile.close();
		std::cout << "Saved: " << fn << std::endl;
	}
	else
	{
		std::cout << "Failed to save: " << fn << std::endl;
	}
}



