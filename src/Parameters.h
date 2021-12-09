#pragma once
#include "common.h"
#include "LF_setting.h"

class Parameters
{
public:
	Parameters();
	Parameters::Parameters(boost::property_tree::ptree pt, LF_setting& lf_setting);
	void Parameters::displayParameters();

	//Parameters from json file
	int disparityResolution;
	float gamma;
	float lambda;
	float sigma;
	int W1;
	int W2;
	int t;
	float mu0_lowres;
	float mu0_highres;
	float kappa;
	float tau;
	std::string viewSelection;
	int usedViewNum;
	bool useOptimization;//enable optimization


};


