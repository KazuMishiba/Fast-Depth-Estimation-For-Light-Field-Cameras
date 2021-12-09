#pragma once

constexpr auto DEFAULT_CONFIG_FILE_NAME = "default.json";//default config file name
constexpr auto LARGE_VALUE = 10000000000.0f;//Value for initialization


struct SceneInfo
{
	int vn;//Number of vertical viewpoints
	int un;//Number of horizontal viewpoints
	int num;//Number of all viewpoints
	int height;//Image height
	int width;//Image width
	float disp_min;//Min disparity of a scene
	float disp_max;//Max disparity of a scene
	//
	int* imageIndex;//Image index for data loader
};
