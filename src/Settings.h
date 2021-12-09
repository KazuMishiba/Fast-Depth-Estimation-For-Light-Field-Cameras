#pragma once
#include "common.h"
#include "Parameters.h"
#include "LF_setting.h"
class Settings
{
public:
	Settings();
	Settings(std::string settingFileName);

	Parameters param;
	LF_setting lf_settings;

	SceneInfo currentSceneInfo;

	bool displayIntermediate;
	bool displayResult;

	// for debug
	struct ReduceData
	{
		bool enable;//Flag for reducing data size of the loading scene
		bool enableLimitView;//Flag for view limitation
		int useViewLength;//view length for view limitation
		bool enableScaleImageSize;//Flag for image scaling
		float scalingRate;//scaling rate
	};
	ReduceData reduceData;


private:
	void loadSetting(boost::property_tree::ptree pt);
};

