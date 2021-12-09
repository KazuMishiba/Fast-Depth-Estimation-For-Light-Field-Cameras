#include "Settings.h"

Settings::Settings(){}

Settings::Settings(std::string settingFileName)
{
	//Read settings from json file
	boost::property_tree::ptree pt;
	std::cout << "Load config file: " << settingFileName << std::endl;
	try
	{
		boost::property_tree::read_json(settingFileName, pt);

		//Setting for image data sets
		this->lf_settings = LF_setting(pt);

		//Setting algorithm parameters
		this->param = Parameters(pt, this->lf_settings);

		//Other settings
		this->loadSetting(pt);
	}
	catch (boost::property_tree::file_parser_error& e)
	{
		std::wcout << L"ERROR : " << e.what() << std::endl;
		exit(1);
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

	//Display parameters
	this->param.displayParameters();

}

//Load settings
void Settings::loadSetting(boost::property_tree::ptree pt)
{
	this->currentSceneInfo = this->lf_settings.sceneInfo;


	std::string baseStr = "ReduceData.";
	this->reduceData.enable = pt.get<int>(baseStr + "enable") == 1;
	if (this->reduceData.enable)
	{
		std::cout << "## Reduce data for debug ##" << std::endl;

		this->reduceData.enableLimitView = (pt.get<int>(baseStr + "enableLimitView") == 1);
		this->reduceData.useViewLength = pt.get<int>(baseStr + "useViewLength");
		this->reduceData.enableScaleImageSize = (pt.get<int>(baseStr + "enableScaleImageSize") == 1);
		this->reduceData.scalingRate = pt.get<float>(baseStr + "scalingRate");

		if (this->reduceData.enableLimitView)
		{
			this->currentSceneInfo.un = this->reduceData.useViewLength;
			this->currentSceneInfo.vn = this->reduceData.useViewLength;
			this->currentSceneInfo.num = this->currentSceneInfo.un * this->currentSceneInfo.vn;

			std::cout << "View Size: " << this->currentSceneInfo.vn << " * " << this->currentSceneInfo.un  << " ( original: " << this->lf_settings.sceneInfo.vn << "*" << this->lf_settings.sceneInfo.un << ")" << std::endl;
		}
		if (this->reduceData.enableScaleImageSize)
		{
			//Size measurement for data reduction
			cv::Mat mat = cv::Mat::zeros(this->currentSceneInfo.height, this->currentSceneInfo.width, CV_32FC1);
			cv::resize(mat, mat, cv::Size(), this->reduceData.scalingRate, this->reduceData.scalingRate, cv::INTER_CUBIC);
			this->currentSceneInfo.width = mat.cols;
			this->currentSceneInfo.height = mat.rows;

			this->currentSceneInfo.disp_min *= this->reduceData.scalingRate;
			this->currentSceneInfo.disp_max *= this->reduceData.scalingRate;

			std::cout << "Image height: " << this->currentSceneInfo.height << " ( original: " << this->lf_settings.sceneInfo.height << ")" << std::endl;
			std::cout << "Image width: " << this->currentSceneInfo.width << " ( original: " << this->lf_settings.sceneInfo.width << ")" << std::endl;
			std::cout << "Min disparity: " << this->currentSceneInfo.disp_min << " ( original: " << this->lf_settings.sceneInfo.disp_min << ")" << std::endl;
			std::cout << "Max disparity: " << this->currentSceneInfo.disp_max << " ( original: " << this->lf_settings.sceneInfo.disp_max << ")" << std::endl;
		}

	}
	else
	{
		this->reduceData.enableLimitView = false;
		this->reduceData.enableScaleImageSize = false;
	}

	if (this->reduceData.enableLimitView) {
		//For view limitation, only the index to be used is extracted and registered.
		std::vector<std::vector<int>> rawIndex;

		rawIndex.resize(this->lf_settings.sceneInfo.vn);
		for (int u = 0; u < this->lf_settings.sceneInfo.un; u++)
		{
			rawIndex[u].resize(this->lf_settings.sceneInfo.un);
		}
		int n = 0;
		for (int v = 0; v < this->lf_settings.sceneInfo.vn; v++)
		{
			for (int u = 0; u < this->lf_settings.sceneInfo.un; u++)
			{
				rawIndex[v][u] = n;
				n++;
			}
		}

		int vc = (this->lf_settings.sceneInfo.vn - 1) / 2;
		int uc = (this->lf_settings.sceneInfo.un - 1) / 2;
		this->currentSceneInfo.imageIndex = new int[this->currentSceneInfo.num];
		n = 0;
		std::cout << "Use Image Index: " << std::endl;
		int vr = (this->currentSceneInfo.vn - 1) / 2;
		int ur = (this->currentSceneInfo.un - 1) / 2;
		for (int v = -vr; v <= vr; v++)
		{

			for (int u = -ur; u <= ur; u++)
			{
				this->currentSceneInfo.imageIndex[n] = rawIndex[vc + v][uc + u];
				std::cout << this->currentSceneInfo.imageIndex[n] << " ";
				n++;
			}
			std::cout << std::endl;
		}
	}
	else {
		//All views are used
		this->currentSceneInfo.imageIndex = new int[this->currentSceneInfo.num];
		for (int n = 0; n < this->currentSceneInfo.num; n++)
		{
			this->currentSceneInfo.imageIndex[n] = n;
		}
	}
	std::cout << std::endl;


	//displayIntermediate
	this->displayIntermediate = pt.get<int>("Debug.displayIntermediate") == 1;
	//displayResult
	this->displayResult = pt.get<int>("Debug.displayResult") == 1;

}
