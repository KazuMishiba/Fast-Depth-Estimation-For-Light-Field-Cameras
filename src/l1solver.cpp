/***************************************************************/
/*
*   The code is created by modifying the code published for the following paper.
*   [1] "100+ Times Faster Weighted Median Filter", Qi Zhang, Li Xu, Jiaya Jia, IEEE Conference on
*		Computer Vision and Pattern Recognition (CVPR), 2014
*
*   The code and the algorithm are for non-comercial use only.
*
/***************************************************************/

#include "l1solver.h"

/***************************************************************/
/* Function: solveWeightedMedian
*
* Description: find xi that minimize K(xi) = \sum_j qj wij |xi - yj| + (mu / pi - qi wij)|xi - yi| + lambda |xi - zi|
*
* input arguments:
*	I: input image. CV_32SC1
*   F: feature image. CV_32SC1
*   r: window radius (W2 - 1) / 2
*	sigma: sigma in Eq. (9).
*   nI: number of possible values in I, i.e., all values of I should in range [0, nI)
*   nF: number of possible values in F, i.e., all values of F should in range [0, nF)
*   p: estimation confidence. CV_32FC1
*	lambda: lambda in Eq. (7).
*	mu0_lowres: \mu_0 in low resolution in coarse-to-fine strategy
*	mu0_highres: \mu_0 in high resolution in coarse-to-fine strategy
*	kappa: parameter \kappa
*	tau: threshold \tau
*/
/***************************************************************/

//Solve optimization problem
cv::Mat l1Solver::solverInterfacePropWithMultiresolution(cv::Mat &I, cv::Mat &F, int r, float sigma, int nI, int nF, cv::Mat &p, float lambda, float mu0_lowres, float mu0_highres, float kappa, float tau) {
	cv::Mat result;
	I.copyTo(result);

	float **wMap;
	featureIndexing(F, wMap, nF, sigma);

	{
		int iter = 0;
		float updatedRate = 1.0;//Rate of pixels that have been updated (1.0=100%)

		cv::Size originalSize = result.size();
		cv::Size lowResSize = { originalSize.width / 2, originalSize.height / 2 };

		//Low resolution
		cv::Mat lowResult, lowI, lowF, lowP;
		cv::resize(result, lowResult, lowResSize, 0, 0, cv::INTER_NEAREST);
		cv::resize(I, lowI, lowResSize, 0, 0, cv::INTER_NEAREST);
		cv::resize(F, lowF, lowResSize, 0, 0, cv::INTER_NEAREST);
		cv::resize(p, lowP, lowResSize, 0, 0, cv::INTER_NEAREST);
		//Window radius (W2 - 1) / 2 for low resolution
		int lowR = std::max(1, int(r / 2));

		float mu = mu0_lowres;

		while (updatedRate > tau)
		{
			lowResult = solveWeightedMedianForIterate(lowResult, lowI, lowF, wMap, lowR, nF, nI, lowP, lambda / 2, mu, updatedRate);
			mu *= kappa;
			iter++;
		}
		std::cout << "Num of iter: " << iter << std::endl;

		//Upsampling
		cv::resize(lowResult, result, originalSize, 0, 0, cv::INTER_NEAREST);
		std::cout << "Upsampled" << std::endl;
		//High(Original) resolution
		updatedRate = 1.0;
		mu = mu0_highres;

		while (updatedRate > tau)
		{
			result = solveWeightedMedianForIterate(result, I, F, wMap, r, nF, nI, p, lambda / 2, mu, updatedRate);
			mu *= kappa;
			iter++;
		}
		std::cout << "Num of iter: " << iter << std::endl;
	}
	float2D_release(wMap);

	return result;
}

//Solver
cv::Mat l1Solver::solveWeightedMedianForIterate(cv::Mat &I, cv::Mat &Z, cv::Mat &F, float **wMap, int r, int nF, int nI, cv::Mat &p, float lambda, float mu, float &updatedRate) {
	// Configuration and declaration
	int rows = I.rows, cols = I.cols;
	int alls = rows * cols;
	int winSize = (2 * r + 1)*(2 * r + 1);
	cv::Mat outImg = I.clone();



	int updateCounter = 0;

	// Column Scanning
#ifdef _DEBUG
#else
#pragma omp parallel for reduction(+:updateCounter) schedule(dynamic, 1)
	//#pragma omp parallel for reduction(+:updateCounter) num_threads(8)
#endif
	for (int x = 0; x<cols; x++) {
		// Allocate memory for joint-histogram and BCB
		int **H = int2D(nI, nF);
		int *BCB = new int[nF];

		// Allocate links for necklace table
		int **Hf = int2D(nI, nF);//forward link
		int **Hb = int2D(nI, nF);//backward link
		int *BCBf = new int[nF];//forward link
		int *BCBb = new int[nF];//backward link

		// Reset histogram and BCB for each column
		memset(BCB, 0, sizeof(int)*nF);
		memset(H[0], 0, sizeof(int)*nF*nI);
		for (int i = 0; i<nI; i++)Hf[i][0] = Hb[i][0] = 0;
		BCBf[0] = BCBb[0] = 0;


		// Reset cut-point
		int medianVal = -1;

		// Precompute "x" range and checks boundary
		int downX = std::max(0, x - r);
		int upX = std::min(cols - 1, x + r);

		// Initialize joint-histogram and BCB for the first window
		{
			int upY = std::min(rows - 1, r);
			for (int i = 0; i <= upY; i++) {

				int *IPtr = I.ptr<int>(i);
				int *FPtr = F.ptr<int>(i);


				for (int j = downX; j <= upX; j++) {
					int fval = IPtr[j];
					int *curHist = H[fval];
					int gval = FPtr[j];

					// Maintain necklace table of joint-histogram
					if (!curHist[gval] && gval) {
						int *curHf = Hf[fval];
						int *curHb = Hb[fval];

						int p1 = 0, p2 = curHf[0];
						curHf[p1] = gval;
						curHf[gval] = p2;
						curHb[p2] = gval;
						curHb[gval] = p1;
					}

					curHist[gval]++;

					// Maintain necklace table of BCB
					updateBCB(BCB[gval], BCBf, BCBb, gval, -1);
				}
			}
		}

		int yi, zi;
		float pi;
		for (int y = 0; y<rows; y++) {
			// Find weighted median with help of BCB and joint-histogram
			{
				yi = I.ptr<int>(y, x)[0];
				zi = Z.ptr<int>(y, x)[0];
				pi = p.ptr<float>(y, x)[0];
				float weightYi = mu - 1;


				float balanceWeight = 0;
				int curIndex = F.ptr<int>(y, x)[0];//feature index of pixel of interest
				float *fPtr = wMap[curIndex];
				int &curMedianVal = medianVal;

				// Compute current balance
				int i = 0;
				do {
					balanceWeight += BCB[i] * fPtr[i];
					i = BCBf[i];
				} while (i);


				int dif1 = curMedianVal - yi;
				balanceWeight += dif1 >= 0 ? weightYi : -weightYi;
				int dif2 = curMedianVal - zi;
				balanceWeight += dif2 >= 0 ? lambda * pi : -lambda * pi;


				// Move cut-point to the left
				if (balanceWeight >= 0) {
					for (; balanceWeight >= 0 && curMedianVal; curMedianVal--) {
						if (curMedianVal == -1)
						{
							continue;
						}
						float curWeight = 0;
						int *nextHist = H[curMedianVal];
						int *nextHf = Hf[curMedianVal];

						// Compute weight change by shift cut-point
						int i = 0;
						do {
							curWeight += (nextHist[i] << 1)*fPtr[i];

							// Update BCB and maintain the necklace table of BCB
							updateBCB(BCB[i], BCBf, BCBb, i, -(nextHist[i] << 1));

							i = nextHf[i];
						} while (i);


						dif1--;
						if (dif1 == -1)
						{
							curWeight = curWeight + weightYi * 2;
						}
						dif2--;
						if (dif2 == -1)
						{
							curWeight = curWeight + lambda * pi * 2;
						}

						balanceWeight -= curWeight;

					}
				}
				// Move cut-point to the right
				else if (balanceWeight < 0) {
					for (; balanceWeight < 0 && curMedianVal != nI - 1; curMedianVal++) {
						float curWeight = 0;
						int *nextHist = H[curMedianVal + 1];
						int *nextHf = Hf[curMedianVal + 1];

						// Compute weight change by shift cut-point
						int i = 0;
						do {
							curWeight += (nextHist[i] << 1)*fPtr[i];

							// Update BCB and maintain the necklace table of BCB
							updateBCB(BCB[i], BCBf, BCBb, i, nextHist[i] << 1);

							i = nextHf[i];
						} while (i);

						dif1++;
						if (dif1 == 0)
						{
							curWeight = curWeight + weightYi * 2;
						}
						dif2++;
						if (dif2 == 0)
						{
							curWeight = curWeight + lambda * pi * 2;
						}

						balanceWeight += curWeight;


					}
				}
				// Weighted median is found and written to the output image
				if (balanceWeight < 0)outImg.ptr<int>(y, x)[0] = curMedianVal + 1;
				else outImg.ptr<int>(y, x)[0] = curMedianVal;

				//update check
				if (yi != outImg.ptr<int>(y, x)[0])
				{
					updateCounter++;
				}


			}

			// Update joint-histogram and BCB when local window is shifted.
			{
				int fval, gval, *curHist;
				// Add entering pixels into joint-histogram and BCB
				{
					int rownum = y + r + 1;
					if (rownum < rows) {
						int *inputImgPtr = I.ptr<int>(rownum);
						int *guideImgPtr = F.ptr<int>(rownum);

						for (int j = downX; j <= upX; j++) {


							fval = inputImgPtr[j];
							curHist = H[fval];
							gval = guideImgPtr[j];

							// Maintain necklace table of joint-histogram
							if (!curHist[gval] && gval) {//
								int *curHf = Hf[fval];
								int *curHb = Hb[fval];
								int p1 = 0, p2 = curHf[0];
								curHf[gval] = p2;
								curHb[gval] = p1;
								curHf[p1] = curHb[p2] = gval;
							}
							curHist[gval]++;

							// Maintain necklace table of BCB
							updateBCB(BCB[gval], BCBf, BCBb, gval, ((fval <= medianVal) << 1) - 1);
						}
					}
				}

				// Delete leaving pixels into joint-histogram and BCB
				{
					int rownum = y - r;
					if (rownum >= 0) {

						int *inputImgPtr = I.ptr<int>(rownum);
						int *guideImgPtr = F.ptr<int>(rownum);

						for (int j = downX; j <= upX; j++) {

							fval = inputImgPtr[j];
							curHist = H[fval];
							gval = guideImgPtr[j];

							curHist[gval]--;

							// Maintain necklace table of joint-histogram
							if (!curHist[gval] && gval) {
								int *curHf = Hf[fval];
								int *curHb = Hb[fval];
								int p1 = curHb[gval], p2 = curHf[gval];
								curHf[p1] = p2;
								curHb[p2] = p1;
							}

							// Maintain necklace table of BCB
							updateBCB(BCB[gval], BCBf, BCBb, gval, -((fval <= medianVal) << 1) + 1);
						}
					}
				}
			}
		}
		// Deallocate the memory
		{
			delete[]BCB;
			delete[]BCBf;
			delete[]BCBb;
			int2D_release(H);
			int2D_release(Hf);
			int2D_release(Hb);

		}

	}
	updatedRate = updateCounter / (float)alls;

	return outImg;
}
