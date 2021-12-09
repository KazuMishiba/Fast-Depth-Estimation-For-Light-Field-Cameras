/***************************************************************/
/*
*   The code is created by modifying the code published for the following paper.
*   [1] "100+ Times Faster Weighted Median Filter", Qi Zhang, Li Xu, Jiaya Jia, IEEE Conference on
*		Computer Vision and Pattern Recognition (CVPR), 2014
*
*   The code and the algorithm are for non-comercial use only.
*
/***************************************************************/

#pragma once

class l1Solver {
public:
	static cv::Mat solverInterfacePropWithMultiresolution(cv::Mat& I, cv::Mat& F, int r, float sigma, int nI, int nF, cv::Mat& p, float lambda, float mu0_lowres, float mu0_highres, float kappa, float tau);
	static cv::Mat solveWeightedMedianForIterate(cv::Mat& I, cv::Mat& Z, cv::Mat& F, float** wMap, int r, int nF, int nI, cv::Mat& p, float lambda, float mu, float& updatedRate);

private:

	/***************************************************************/
	/* Function: updateBCB
	* Description: maintain the necklace table of BCB
	/***************************************************************/
	static inline void updateBCB(int& num, int* f, int* b, int i, int v) {

		static int p1, p2;
		if (i) {
			if (!num) { // cell is becoming non-empty
				p2 = f[0];
				f[0] = i;
				f[i] = p2;
				b[p2] = i;
				b[i] = 0;
			}
			else if (!(num + v)) {// cell is becoming empty
				p1 = b[i], p2 = f[i];
				f[p1] = p2;
				b[p2] = p1;
			}
		}
		// update the cell count
		num += v;
	}

	/***************************************************************/
	/* Function: float2D
	* Description: allocate a 2D float array with dimension "dim1 x dim2"
	/***************************************************************/
	static float** float2D(int dim1, int dim2) {
		float** ret = new float* [dim1];
		ret[0] = new float[dim1 * dim2];
		for (int i = 1; i < dim1; i++)ret[i] = ret[i - 1] + dim2;

		return ret;
	}

	/***************************************************************/
	/* Function: float2D_release
	* Description: deallocate the 2D array created by float2D()
	/***************************************************************/
	static void float2D_release(float** p) {
		delete[]p[0];
		delete[]p;
	}

	/***************************************************************/
	/* Function: int2D
	* Description: allocate a 2D integer array with dimension "dim1 x dim2"
	/***************************************************************/
	static int** int2D(int dim1, int dim2) {
		int** ret = new int* [dim1];
		ret[0] = new int[dim1 * dim2];
		for (int i = 1; i < dim1; i++)ret[i] = ret[i - 1] + dim2;

		return ret;
	}

	/***************************************************************/
	/* Function: int2D_release
	* Description: deallocate the 2D array created by int2D()
	/***************************************************************/
	static void int2D_release(int** p) {
		delete[]p[0];
		delete[]p;
	}

	/***************************************************************/
	/* Function: featureIndexing
	* Description: Computer weight map (weight between each pair of feature index)
	/***************************************************************/
	static void featureIndexing(cv::Mat& F, float**& wMap, int& nF, float sigmaI) {
		wMap = float2D(nF, nF);
		float nSigmaI = sigmaI;
		float divider = (1.0f / (2 * nSigmaI * nSigmaI));

		for (int i = 0; i < nF; i++) {
			for (int j = i; j < nF; j++) {
				float diff = fabs((float)(i - j));
				wMap[i][j] = wMap[j][i] = exp(-(diff * diff) * divider); // EXP 2
			}
		}
	}

};
