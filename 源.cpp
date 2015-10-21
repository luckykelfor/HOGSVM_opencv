#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include<io.h>
#include <ios>
#include <fstream>
using namespace std;
using namespace cv;



#define YES 1
#define  NO 0
#define DESCRIPTOR_FIRST_SAVED YES
#define DESCRIPTOR_SECOND_SAVED YES
#define HARD_SAMPLE_SAVED YES
#define CENTRAL_CROP false  //true:训练时，对96*160的INRIA正样本图片剪裁出中间的64*128大小人体
#define RANDNEG_SAVED YES


#define WINSIZE_HEIGHT 128
#define WINSIZE_WIDTH 64


// 存储特征描述
static string featuresFile = "D:/Myworkspace/ImageData/BottlesSet/features_opencvVersion_bottle.dat";

// 存储HOG特征描述符
static string descriptorVectorFile = "D:/Myworkspace/ImageData/BottlesSet/descriptorvector_opencvVersion_bottle.dat";

//original 训练样本路径名
static string posTrainingSamplesDir = "D:/Myworkspace/ImageData/BottlesSet/pos/";
static string originalNegTrainingSamplesDir = "D:/Myworkspace/ImageData/BottlesSet/neg/";
//从原始负样本随机裁剪的图像
static string randNegSamplesDir = "D:/Myworkspace/ImageData/BottlesSet/randneg/";

//测试样本路径名
static string testPosDir = "D:/Myworkspace/ImageData/BottlesSet/test/pos/";
static string testNegDir = "D:/Myworkspace/ImageData/BottlesSet/test/neg/";
static string testSamplesDir = "D:/Myworkspace/ImageData/BottlesSet/test/";
//生成的难例存储路径：
static string hardSampleDir = "D:/Myworkspace/ImageData/BottlesSet/hard_opencvVersion/";

static const Size trainingPadding = Size(0, 0); //填充值
static const Size winStride = Size(8, 8);			//窗口步进

													//保存最后的SVM模型
static string svmModelFile_first = "D:/Myworkspace/ImageData/BottlesSet/svmModel_first.xml";
static string svmModelFile = "D:/Myworkspace/ImageData/BottlesSet/svmModel.xml";


//继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，
//但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问
class MySVM : public CvSVM
{
public:
	//获得SVM的决策函数中的alpha数组
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//获得SVM的决策函数中的rho参数,即偏移量
	float get_rho()
	{
		return this->decision_func->rho;
	}
};

/************************************************************************/
/* 保存计算出来的特征描述*/
/************************************************************************/
static void saveFeatures(const string saveFeatureFileName, const Mat& sampleFeatures, const Mat& sampleLabel)
{
	ofstream writeFeatureFile(saveFeatureFileName.c_str(), ios::out);
	if (writeFeatureFile.good() && writeFeatureFile.is_open())
	{
		for (int i = 0; i<sampleFeatures.rows; i++)
		{
			writeFeatureFile << sampleLabel.at<float>(i) << " ";
			for (int j = 0; j<sampleFeatures.cols; j++)
			{
				writeFeatureFile << sampleFeatures.at<float>(i, j) << " ";
			}
			writeFeatureFile << endl;

		}
	}
	else
	{
		return;
	}
	writeFeatureFile.close();
	cout << "Save FeatureMat Done.\n";

}


/************************************************************************/
/* 直接从文件加载已经保存好的特征描述（用at访问Mat元素很慢）                                                                     */
/************************************************************************/
/*This version of loading file is two slow(using the at method*/
// static void loadFeatures(const string featureFileName, Mat& featureMat, Mat &labelMat)
// {
// 	ifstream readFeature(featureFileName.c_str(), ios::in);
// 	 int rows = featureMat.rows;
// 	 int cols = featureMat.cols;
// 	for (int i = 0; i < rows;i++)
// 	{
// 		readFeature>>labelMat.at<float>(i);
// 		for (int j = 0; j < cols;j++)
// 		{
// 			readFeature>>featureMat.at<float>(i, j);
// 		}
// 	}
// 	readFeature.close();
// 	cout << "Read Features Done.\n";
// }
static void loadFeatures(const string featureFileName, Mat& featureMat, Mat &labelMat)
{
	ifstream readFeature(featureFileName.c_str(), ios::in);
	int rows = featureMat.rows;
	int cols = featureMat.cols;
	float * p;
	for (int i = 0; i<rows; i++)
	{
		p = featureMat.ptr<float>(i);
		for (int j = 0; j<cols; j++)
		{
			readFeature >> p[j];
		}
	}
}

/************************************************************************/
/* 训练SVM
input: criteria, param是SVM参数设置 myDetector是传入的待训练的描述子Descriptor, sampleFeatureMat是
用于训练的样本的特征 sampleLabelMat是对应的分类情况。*/
/************************************************************************/
static void getTrainedDetector(MySVM&  svm, CvTermCriteria & criteria, CvSVMParams &param, vector<float>&myDetector, Mat &sampleFeatureMat, Mat& sampleLabelMat)
{
	//训练SVM分类器 第一次训练 没有加入 HardSamples的训练
	//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代

	//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01



	/*************************************************************************************************
	线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
	将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量。之后，再该列向量的最后添加一个元素rho。
	如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
	就可以利用你的训练样本训练出来的分类器进行行人检测了。
	***************************************************************************************************/
	cout << "开始训练SVM分类器" << endl;
	svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器
	cout << "训练完成，保存模型到文件" << endl;
	svm.save(svmModelFile_first.c_str());//将训练好的SVM模型保存为xml文件
	int DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
	cout << "支持向量个数：" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果

														   //将支持向量的数据复制到supportVectorMat矩阵中
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}



	//将alpha向量的数据复制到alphaMat中
	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	//将resultMat中的数据复制到数组myDetector中
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm.get_rho());
	cout << "检测子维数：" << myDetector.size() << endl;
	cout << "偏移量rho=" << svm.get_rho() << endl;
}







/**
* 保存给定的HOG特征描述符到文件
* @param descriptorVector: 待保存的HOG特征描述符矢量
* @param _vectorIndices
* @param fileName
*/
static void saveDescriptorVectorToFile(vector<float>& descriptorVector, string fileName) {
	printf("保存HOG特征描述符：'%s'\n", fileName.c_str());
	string separator = " "; // 特征分隔符
	fstream File;
	File.open(fileName.c_str(), ios::out);
	if (File.good() && File.is_open()) {
		for (int feature = 0; feature < descriptorVector.size(); ++feature)
			File << descriptorVector.at(feature) << separator;	//写入特征并设置分隔符号
																// File << endl;
		File.flush();
		File.close();
	}
}

/************************************************************************/
/* 从文件中读取保存的Descriptor描述子                                                                     */
/************************************************************************/
static void 	loadDescriptorFromFile(vector<float>& myDescriptor, const string descriptorVectorFile)
{
	ifstream loadDescriptor(descriptorVectorFile, ios::in);
	float temp = 0;
	if (loadDescriptor.good() && loadDescriptor.is_open())
	{

		while (!loadDescriptor.eof())
		{
			loadDescriptor >> temp;
			myDescriptor.push_back(temp);
		}
	}
	else
	{
		cout << "Loading Descriptor failed.\n";

	}
}

/************************************************************************/
/* 在测试集上进行验证，对检测子的性能只是一个估计，没有必然的联系                                                                     */
/************************************************************************/
static void detecOnSet(const HOGDescriptor& hog, const double hitThreshold, const vector<string>& posFileNames, const vector<string>& negFileNames) {
	unsigned int truePositives = 0;
	unsigned int trueNegatives = 0;
	unsigned int falsePositives = 0;
	unsigned int falseNegatives = 0;
	vector<Point> foundDetection;
	// Walk over positive training samples, generate images and detect
	for (vector<string>::const_iterator posTrainingIterator = posFileNames.begin(); posTrainingIterator != posFileNames.end(); ++posTrainingIterator) {
		const Mat imageData = imread(*posTrainingIterator);
		hog.detect(imageData, foundDetection, abs(hitThreshold), winStride, trainingPadding);
		if (foundDetection.size() > 0) {
			++truePositives;
			falseNegatives += foundDetection.size() - 1;
		}
		else {
			++falseNegatives;
		}
	}
	// Walk over negative training samples, generate images and detect
	for (vector<string>::const_iterator negTrainingIterator = negFileNames.begin(); negTrainingIterator != negFileNames.end(); ++negTrainingIterator) {
		const Mat imageData = imread(*negTrainingIterator);
		hog.detect(imageData, foundDetection, abs(hitThreshold), winStride, trainingPadding);
		if (foundDetection.size() > 0) {
			falsePositives += foundDetection.size();
		}
		else {
			++trueNegatives;
		}
	}

	printf("Results:\n\tTrue Positives: %u\n\tTrue Negatives: %u\n\tFalse Positives: %u\n\tFalse Negatives: %u\n", truePositives, trueNegatives, falsePositives, falseNegatives);
}

/************************************************************************/
/* 在测试集上进行检测                                                                     */
/************************************************************************/
static void detectTest(HOGDescriptor &hog, float hitThreshold, vector<string> & testSamples)
{
	Mat src;
	vector<Rect> found, found_filtered;//矩形框数组
	for (vector<string>::iterator i = testSamples.begin(); i < testSamples.end(); i++)
	{

		src = imread(*i);

		cout << "进行多尺度HOG人体检测" << endl;
		hog.detectMultiScale(src, found, abs(hitThreshold),Size(8,8), Size(0, 0), 1.15, 2);//对图片进行多尺度行人检测
		cout << "找到的矩形框个数：" << found.size() << endl;

		//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
		for (int i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			int j = 0;
			for (; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}

		//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
		for (int i = 0; i<found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(src, r.tl(), r.br(), Scalar(0, 255, 0), 3);
		}

		found_filtered.clear();
		found.clear();
		imshow("src", src);
		waitKey(500);//注意：imshow之后必须加waitKey，否则无法显示图像
	}

}

/************************************************************************/
/* 转小写函数                                                                     */
/************************************************************************/
static string toLowerCase(const string& in) {
	string t;
	for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
		t += tolower(*i);
	}
	return t;
}



/************************************************************************/
/* 从目录生成生成样本的完整文件名数组                                                                      */
/************************************************************************/
static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
	printf("扫描样本目录 %s\n", dirName.c_str());
	long hFile = 0;
	struct _finddata_t fileInfo;
	string pathName, fullfileName;
	string  tempPathName = pathName.assign(dirName);
	string  tempPathName2 = pathName.assign(dirName);

	//if ((hFile = _findfirst(pathName.assign(dirName).append("\\*").c_str(), &fileInfo)) == -1) {
	if ((hFile = _findfirst(tempPathName.append("\\*").c_str(), &fileInfo)) == -1) {
		return;
	}
	do {
		if (fileInfo.attrib&_A_SUBDIR)//文件夹跳过
			continue;
		else
		{

			int i = string(fileInfo.name).find_last_of(".");
			string tempExt = toLowerCase(string(fileInfo.name).substr(i + 1));

			if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end())
			{


				fullfileName = tempPathName2 + (string(fileInfo.name));
				//  cout<<"有效文件： '%s'\n"<< fullfileName << endl;
				fileNames.push_back((cv::String)fullfileName);



			}
			else
			{
				cout << "不是图像文件，跳过: '%s'\n" << fileInfo.name << endl;
			}


		}

	} while (_findnext(hFile, &fileInfo) == 0);
	_findclose(hFile);
	return;

}




/************************************************************************/
/* 生成难例：在原训练集上检测，将误报的方框大小保存为难例，加入原始的负样本
集中，供给第二次训练使用*/
/************************************************************************/
static long findHardExmaple(const HOGDescriptor& hog, const double hitThreshold, const vector<string>& negFileNames, const string hardExampleDir)
{

	//在Neg中检测出来的都是误报的，将多尺度检测出来的窗口作为难例。
	char saveName[256];
	long hardExampleCount = 0;
	vector<Rect> foundDetection;
	// Walk over negative training samples, generate images and detect
	for (vector<string>::const_iterator negTrainingIterator = negFileNames.begin(); negTrainingIterator != negFileNames.end(); ++negTrainingIterator)
	{
		const Mat imageData = imread(*negTrainingIterator);
		hog.detectMultiScale(imageData, foundDetection, hitThreshold, winStride, Size(32,32));
		//遍历从图像中检测出来的矩形框，得到hard example
		for (int i = 0; i < foundDetection.size(); i++)
		{
			//检测出来的很多矩形框都超出了图像边界，将这些矩形框都强制规范在图像边界内部
			Rect r = foundDetection[i];
			if (r.x < 0)
				r.x = 0;
			if (r.y < 0)
				r.y = 0;
			if (r.x + r.width > imageData.cols)
				r.width = imageData.cols - r.x;
			if (r.y + r.height > imageData.rows)
				r.height = imageData.rows - r.y;

			//将矩形框保存为图片，就是Hard Example
			Mat hardExampleImg = imageData(r);//从原图上截取矩形框大小的图片
			resize(hardExampleImg, hardExampleImg, Size(WINSIZE_WIDTH, WINSIZE_HEIGHT));//将剪裁出来的图片缩放为64*128大小
			sprintf_s(saveName, "hardexample%09ld.jpg", hardExampleCount++);//生成hard example图片的文件名
			imwrite(hardExampleDir + saveName, hardExampleImg);//保存文件
		}

	}
	printf("Hard Example Detected Done.\n");
	return hardExampleCount;


}







/************************************************************************/
/* 主函数                                                                     */
/************************************************************************/



int main()
{
	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9

	HOGDescriptor hog(Size(WINSIZE_WIDTH, WINSIZE_HEIGHT), Size(16, 16), Size(8,8), Size(8,8), 9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器
	HOGDescriptor myHOG(Size(WINSIZE_WIDTH,WINSIZE_HEIGHT), Size(16, 16), Size(8, 8), Size(8, 8), 9);;

	/************************************************************************/
	/* 生成训练正负样本，测试正负样本等的完整路径                                                                     */
	/************************************************************************/
	static vector<string> validExtensions;
	validExtensions.push_back("jpg");
	validExtensions.push_back("png");
	validExtensions.push_back("ppm");
	
	vector<string> testSamples,
		testNegSamples,
		testPosSamples,
		posTrainingSamples,
		originalNegTrainingSamples,
		hardSamples,
		randNegTrainingSamples;

	/*扫描目录得到样本图像完整路径和名称*/
	getFilesInDirectory(testSamplesDir, testSamples, validExtensions);
	getFilesInDirectory(posTrainingSamplesDir, posTrainingSamples, validExtensions);
	getFilesInDirectory(originalNegTrainingSamplesDir, originalNegTrainingSamples, validExtensions);
	getFilesInDirectory(testNegDir, testNegSamples, validExtensions);
	getFilesInDirectory(testPosDir, testPosSamples, validExtensions);



	//Randomly generate 10 negative sample of size 64*128 for each original negative sample in /neg folder. Totally we weill ge 1,2180 neg samples

	////图片大小应该能能至少包含一个64*128的窗口
	if (RANDNEG_SAVED == NO)
	{
		char saveName[256];
		long CropImageCount = 0;
		for (vector<string>::iterator i = originalNegTrainingSamples.begin(); i < originalNegTrainingSamples.end(); i++)
		{
			Mat src = imread(*i);
			if (src.cols >= 128 && src.rows >= 128)
			{
				//srand(time(NULL));//设置随机数种子

				//从每张图片中随机裁剪10个64*128大小的不包含人的负样本
				for (int i = 0; i < 10; i++)
				{
					int x = (rand() % (src.cols - 128)); //左上角x坐标
					int y = (rand() % (src.rows - 128)); //左上角y坐标
														 //cout<<x<<","<<y<<endl;
					Mat imgROI = src(Rect(x, y, WINSIZE_WIDTH, WINSIZE_HEIGHT));
					sprintf_s(saveName, "noperson%06d.jpg", ++CropImageCount);//生成裁剪出的负样本图片的文件名
					imwrite(randNegSamplesDir + saveName, imgROI);//保存文件
				}
			}
		}
	}
	

	getFilesInDirectory(randNegSamplesDir, randNegTrainingSamples, validExtensions);



	long int totalTrainingSampleNO = posTrainingSamples.size() + randNegTrainingSamples.size();
	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数	
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人


#if DESCRIPTOR_FIRST_SAVED == NO
	int num = 0;
	ofstream writeFeatureToFile(featuresFile.c_str(), ios::out);
	if (!(writeFeatureToFile.good() && writeFeatureToFile.is_open()))
	{
		cout << "Cannot open file to save features.\n";
	}

	cout << "处理正样本" << endl;
	for (vector<string>::iterator iter = posTrainingSamples.begin(); iter<posTrainingSamples.end(); iter++, num++)
	{
		//out << "处理：" << *iter << endl;
		Mat src = imread(*iter);//读取图片
		if (CENTRAL_CROP)
			if (src.size() == Size(96, 160))
				src = src(Rect(16, 16, 64, 128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
			else
				resize(src, src, Size(64, 128));

		vector<float> descriptors;//HOG描述子向量
		hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
												  //cout<<"描述子维数："<<descriptors.size()<<endl;

												  //存储Feature到文件
		writeFeatureToFile << 1 << " ";//存储类别+1表示有行人，-1表示没有行人
		for (vector<float>::iterator it = descriptors.begin(); it < descriptors.end(); it++)
		{
			writeFeatureToFile << *it << " ";
		}

		//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
		if (posTrainingSamples.begin() == iter)
		{
			DescriptorDim = descriptors.size();//HOG描述子的维数
											   //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat

			sampleFeatureMat = Mat::zeros(totalTrainingSampleNO, DescriptorDim, CV_32FC1);
			//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
			sampleLabelMat = Mat::zeros(totalTrainingSampleNO, 1, CV_32FC1);
		}

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for (int i = 0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素
		sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人
	}

	//依次读取负样本图片，生成HOG描述子
	cout << "处理负样本" << endl;
	num = 0;
	for (vector<string>::iterator iter = randNegTrainingSamples.begin(); iter<randNegTrainingSamples.end(); iter++, num++)
	{
		cout << "处理：" << *iter << endl;
		//ImgName = "D:\\DataSet\\NoPersonFromINRIA\\" + ImgName;//加上负样本的路径名
		Mat src = imread(*iter);//读取图片
		vector<float> descriptors;//HOG描述子向量
		hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
												  //cout<<"描述子维数："<<descriptors.size()<<endl;
		writeFeatureToFile << -1 << " ";//存储类别+1表示有行人，-1表示没有行人
		for (vector<float>::iterator it = descriptors.begin(); it < descriptors.end(); it++)
		{
			writeFeatureToFile << *it << " ";
		}

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for (int i = 0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(num + posTrainingSamples.size(), i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
		sampleLabelMat.at<float>(num + posTrainingSamples.size(), 0) = -1;//负样本类别为-1，无人
	}


	writeFeatureToFile.close();//关闭文件

							   //训练SVM分类器 第一次训练 没有加入 HardSamples的训练
							   //迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
	vector<float> myDetector;
	getTrainedDetector(svm, criteria, param, myDetector, sampleFeatureMat, sampleLabelMat);
	saveDescriptorVectorToFile(myDetector, descriptorVectorFile);
	float temp_rho = myDetector.at(myDetector.size()-1);
	myHOG.setSVMDetector(myDetector);
	cout << "the rho is:" << temp_rho << endl;
	//以上，第一次训练结束, 得到描述子



#else
	vector<float> descriptors;//HOG描述子向量,只是为了得到描述子的维数，没有任何意义;
	hog.compute(imread(randNegTrainingSamples.at(1).c_str()), descriptors, Size(8, 8));
	DescriptorDim = descriptors.size();
	sampleLabelMat = Mat::zeros(totalTrainingSampleNO, 1, CV_32FC1);
	sampleFeatureMat = Mat::zeros(totalTrainingSampleNO, descriptors.size(), CV_32FC1);

	if (DESCRIPTOR_SECOND_SAVED == NO)
	{
		loadFeatures(featuresFile, sampleFeatureMat, sampleLabelMat);

		svm.load(svmModelFile_first.c_str());
	}
	//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
	vector<float> myDetector;
	loadDescriptorFromFile(myDetector, descriptorVectorFile);
	myDetector.pop_back();//去掉最后一个重复的值
						  //迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	//SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
	float temp_rho = 0.0;
	myHOG.setSVMDetector(myDetector);
#endif
	cout << "在训练集上进行行人检测(结果仅作参考)\n";
	detecOnSet(myHOG, 0, posTrainingSamples, originalNegTrainingSamples);

#if DESCRIPTOR_SECOND_SAVED == NO

	//得到hardSamples列表
	//生成难例并保存
	cout << "Generating hard samples and save them...\n";
#if HARD_SAMPLE_SAVED == NO
	findHardExmaple(myHOG, 0, originalNegTrainingSamples, hardSampleDir);
#endif //#if HARD_SAMPLE_SAVED
	getFilesInDirectory(hardSampleDir, hardSamples, validExtensions);

	//处理HardSample负样本  , HardSamples已经裁剪成64x128大小
	//long int hardSampleNO = hardSamples.size();
	cout << "Adding  hard samples to training....\n";
	Mat hardSampleFeatureMat = Mat::zeros(hardSamples.size(), DescriptorDim, CV_32FC1);
	Mat hardSampleLabelMat = Mat::zeros(hardSamples.size(), 1, CV_32FC1);
	if (hardSamples.size() > 0)
	{
		ofstream addFeature(featuresFile.c_str(), ios::app);

		cout << "处理难例。。。\n";
		int num = 0;															 //依次读取HardExample负样本图片，生成HOG描述子
		for (vector<string>::iterator iter = hardSamples.begin(); iter<hardSamples.end(); iter++, num++)
		{
			/*cout << "处理难例：" << *iter << endl;*/
			//	ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//加上HardExample负样本的路径名
			Mat src = imread(*iter);//读取图片

			vector<float> descriptors;//HOG描述子向量
			hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)
													  //将计算好的HOG描述子复制到样本特征矩阵hardsampleFeatureMat

													  //存储Feature到文件
			addFeature << -1 << " ";//存储类别+1表示有行人，-1表示没有行人
			for (vector<float>::iterator it = descriptors.begin(); it < descriptors.end(); it++)
			{
				addFeature << *it << " ";
			}

			for (int i = 0; i<descriptors.size(); i++)
				hardSampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素
			hardSampleLabelMat.at<float>(num, 0) = -1;//负样本类别为-1，无人
		}//end for

		addFeature.close();
	}//end if


	 //合并
	cout << "Merging the FeatureMat...\n";
	sampleFeatureMat.push_back(hardSampleFeatureMat);//合并
	sampleLabelMat.push_back(hardSampleLabelMat);



	//第二次训练，训练参数保持不变；
	myDetector.clear();
	getTrainedDetector(svm, criteria, param, myDetector, sampleFeatureMat, sampleLabelMat);
	//再次保存Descriptor
	saveDescriptorVectorToFile(myDetector, descriptorVectorFile);
	temp_rho = myDetector.at(myDetector.size()-1);
	cout << "the second time, rho is: " << temp_rho << endl;
	myHOG.setSVMDetector(myDetector);

	svm.save(svmModelFile.c_str());//将训练好的SVM模型保存为xml文件  


								   /**************读入图片进行HOG行人检测******************/
								   //detectTest(myHOG, 0, testSamples);
	cout << "在训练集上进行行人检测(结果仅作参考)\n";

#endif //#if DESCRIPTOR_SECOND_SAVED
	detecOnSet(myHOG, 0, posTrainingSamples,originalNegTrainingSamples);
	detectTest(myHOG, 0, testSamples);
	waitKey(1000);
	return 0;
}