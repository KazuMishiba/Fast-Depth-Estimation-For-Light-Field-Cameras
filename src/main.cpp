#include "main.h"

int main(int argc, char *argv[])
{
	try
	{
		std::vector<std::string> configFileNames;
		if (argc >= 2)
		{
			//Whether the argument is a directory or a file
			bool isDirectory = std::filesystem::is_directory(argv[1]);
			if (isDirectory)
			{
				//Read all .json files in the specified directory as configuration files
				std::filesystem::directory_iterator e = std::filesystem::directory_iterator(argv[1]);
				for (auto f : e) {
					if (f.path().extension() == ".json")
					{
						configFileNames.push_back(f.path().string());
					}
				}
			}
			else
			{
				//Read argv[1] as a configuration file
				configFileNames.push_back(argv[1]);
			}
		}
		else
		{
			//Load DEFAULT_CONFIG_FILE_NAME located in the directory where the EXE is located.
			configFileNames.push_back(DEFAULT_CONFIG_FILE_NAME);
		}

		for (int i = 0; i < configFileNames.size(); i++)
		{
			std::cout << "#####################################################" << std::endl;
			std::cout << "Start disparity estimation (" << i + 1 << "/" << configFileNames.size() << ")" << std::endl;
			std::cout << "#####################################################" << std::endl;
			std::string configFileName = configFileNames[i];

			std::filesystem::path p = std::filesystem::path(configFileName);
			if (p.is_relative())
			{
				//Convert to absolute path
				configFileName = std::filesystem::absolute(p).string();
			}
			//Setting
			Settings settings = Settings(configFileName);

			//Perform
			Prop prop = Prop(settings);
			if (settings.lf_settings.estimateAll)
			{
				//Estimate disparity for all the viewpoints
				int num = prop.numOfAllView();
				prop.perform(0, true);
				for (int i = 1; i < num; i++)
				{
					prop.perform(i, false);
				}
			}
			else
			{
				if (settings.lf_settings.targetIndex == -1)
				{
					//Estimate disparity for the center viewpoint
					prop.perform(prop.centerIndex());
				}
				else
				{
					//Estimate disparity for a specified viewpoint
					prop.perform(settings.lf_settings.targetIndex);
				}
			}
		}

		cudaDeviceReset();

	}
	catch (...)
	{
		std::cerr << "Error! An unknow type of exception occurred." << std::endl;

		cudaDeviceReset();
		exit(EXIT_FAILURE);
		return -1;
	}

}