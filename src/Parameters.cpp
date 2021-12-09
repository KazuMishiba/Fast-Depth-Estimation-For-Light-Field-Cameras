#include "Parameters.h"

Parameters::Parameters(){}

Parameters::Parameters(boost::property_tree::ptree pt, LF_setting& lf_setting)
{
	//Parameter
	this->disparityResolution = pt.get<int>("Parameter.disparityResolution");
	this->gamma = pt.get<float>("Parameter.gamma");
	this->lambda = pt.get<float>("Parameter.lambda");
	this->sigma = pt.get<float>("Parameter.sigma");
	this->W1 = pt.get<int>("Parameter.W1");
	this->W2 = pt.get<int>("Parameter.W2");
	this->t = pt.get<int>("Parameter.t");
	this->mu0_lowres = pt.get<float>("Parameter.mu0_lowres");
	this->mu0_highres = pt.get<float>("Parameter.mu0_highres");
	this->kappa = pt.get<float>("Parameter.kappa");
	this->tau = pt.get<float>("Parameter.tau");
	this->viewSelection = pt.get<std::string>("Parameter.viewSelection");
	this->usedViewNum = pt.get<int>("Parameter.usedViewNum");
	this->useOptimization = (pt.get<int>("Parameter.useOptimization") == 1);
}

void Parameters::displayParameters() {
	std::cout << "## Parameters ##" << std::endl;
	std::cout << "Alpha_max: " << this->disparityResolution << std::endl;
	std::cout << "gamma: " << this->gamma << std::endl;
	std::cout << "lambda: " << this->lambda << std::endl;
	std::cout << "sigma: " << this->sigma << std::endl;
	std::cout << "W1: " << this->W1 << std::endl;
	std::cout << "W2: " << this->W2 << std::endl;
	std::cout << "t (disparity sampling rate): " << this->t << std::endl;
	std::cout << "mu0 (low res): " << this->mu0_lowres << std::endl;
	std::cout << "mu0 (high res): " << this->mu0_highres << std::endl;
	std::cout << "kappa: " << this->kappa << std::endl;
	std::cout << "tau: " << this->tau << std::endl;
	std::cout << "view selection: " << this->viewSelection << std::endl;
	std::cout << "used view num: " << this->usedViewNum << std::endl;
	std::cout << "Use optimization: " << this->useOptimization << std::endl << std::endl;


}

