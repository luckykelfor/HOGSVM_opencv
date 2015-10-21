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
#define CENTRAL_CROP false  //true:ѵ��ʱ����96*160��INRIA������ͼƬ���ó��м��64*128��С����
#define RANDNEG_SAVED YES


#define WINSIZE_HEIGHT 128
#define WINSIZE_WIDTH 64


// �洢��������
static string featuresFile = "D:/Myworkspace/ImageData/BottlesSet/features_opencvVersion_bottle.dat";

// �洢HOG����������
static string descriptorVectorFile = "D:/Myworkspace/ImageData/BottlesSet/descriptorvector_opencvVersion_bottle.dat";

//original ѵ������·����
static string posTrainingSamplesDir = "D:/Myworkspace/ImageData/BottlesSet/pos/";
static string originalNegTrainingSamplesDir = "D:/Myworkspace/ImageData/BottlesSet/neg/";
//��ԭʼ����������ü���ͼ��
static string randNegSamplesDir = "D:/Myworkspace/ImageData/BottlesSet/randneg/";

//��������·����
static string testPosDir = "D:/Myworkspace/ImageData/BottlesSet/test/pos/";
static string testNegDir = "D:/Myworkspace/ImageData/BottlesSet/test/neg/";
static string testSamplesDir = "D:/Myworkspace/ImageData/BottlesSet/test/";
//���ɵ������洢·����
static string hardSampleDir = "D:/Myworkspace/ImageData/BottlesSet/hard_opencvVersion/";

static const Size trainingPadding = Size(0, 0); //���ֵ
static const Size winStride = Size(8, 8);			//���ڲ���

													//��������SVMģ��
static string svmModelFile_first = "D:/Myworkspace/ImageData/BottlesSet/svmModel_first.xml";
static string svmModelFile = "D:/Myworkspace/ImageData/BottlesSet/svmModel.xml";


//�̳���CvSVM���࣬��Ϊ����setSVMDetector()���õ��ļ���Ӳ���ʱ����Ҫ�õ�ѵ���õ�SVM��decision_func������
//��ͨ���鿴CvSVMԴ���֪decision_func������protected���ͱ������޷�ֱ�ӷ��ʵ���ֻ�ܼ̳�֮��ͨ����������
class MySVM : public CvSVM
{
public:
	//���SVM�ľ��ߺ����е�alpha����
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//���SVM�ľ��ߺ����е�rho����,��ƫ����
	float get_rho()
	{
		return this->decision_func->rho;
	}
};

/************************************************************************/
/* ��������������������*/
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
/* ֱ�Ӵ��ļ������Ѿ�����õ�������������at����MatԪ�غ�����                                                                     */
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
/* ѵ��SVM
input: criteria, param��SVM�������� myDetector�Ǵ���Ĵ�ѵ����������Descriptor, sampleFeatureMat��
����ѵ�������������� sampleLabelMat�Ƕ�Ӧ�ķ��������*/
/************************************************************************/
static void getTrainedDetector(MySVM&  svm, CvTermCriteria & criteria, CvSVMParams &param, vector<float>&myDetector, Mat &sampleFeatureMat, Mat& sampleLabelMat)
{
	//ѵ��SVM������ ��һ��ѵ�� û�м��� HardSamples��ѵ��
	//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����

	//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01



	/*************************************************************************************************
	����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho;
	��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ����������֮���ٸ���������������һ��Ԫ��rho��
	��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()����
	�Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�
	***************************************************************************************************/
	cout << "��ʼѵ��SVM������" << endl;
	svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//ѵ��������
	cout << "ѵ����ɣ�����ģ�͵��ļ�" << endl;
	svm.save(svmModelFile_first.c_str());//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�
	int DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���
	cout << "֧������������" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��

														   //��֧�����������ݸ��Ƶ�supportVectorMat������
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}



	//��alpha���������ݸ��Ƶ�alphaMat��
	double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm.get_rho());
	cout << "�����ά����" << myDetector.size() << endl;
	cout << "ƫ����rho=" << svm.get_rho() << endl;
}







/**
* ���������HOG�������������ļ�
* @param descriptorVector: �������HOG����������ʸ��
* @param _vectorIndices
* @param fileName
*/
static void saveDescriptorVectorToFile(vector<float>& descriptorVector, string fileName) {
	printf("����HOG������������'%s'\n", fileName.c_str());
	string separator = " "; // �����ָ���
	fstream File;
	File.open(fileName.c_str(), ios::out);
	if (File.good() && File.is_open()) {
		for (int feature = 0; feature < descriptorVector.size(); ++feature)
			File << descriptorVector.at(feature) << separator;	//д�����������÷ָ�����
																// File << endl;
		File.flush();
		File.close();
	}
}

/************************************************************************/
/* ���ļ��ж�ȡ�����Descriptor������                                                                     */
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
/* �ڲ��Լ��Ͻ�����֤���Լ���ӵ�����ֻ��һ�����ƣ�û�б�Ȼ����ϵ                                                                     */
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
/* �ڲ��Լ��Ͻ��м��                                                                     */
/************************************************************************/
static void detectTest(HOGDescriptor &hog, float hitThreshold, vector<string> & testSamples)
{
	Mat src;
	vector<Rect> found, found_filtered;//���ο�����
	for (vector<string>::iterator i = testSamples.begin(); i < testSamples.end(); i++)
	{

		src = imread(*i);

		cout << "���ж�߶�HOG������" << endl;
		hog.detectMultiScale(src, found, abs(hitThreshold),Size(8,8), Size(0, 0), 1.15, 2);//��ͼƬ���ж�߶����˼��
		cout << "�ҵ��ľ��ο������" << found.size() << endl;

		//�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
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

		//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
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
		waitKey(500);//ע�⣺imshow֮������waitKey�������޷���ʾͼ��
	}

}

/************************************************************************/
/* תСд����                                                                     */
/************************************************************************/
static string toLowerCase(const string& in) {
	string t;
	for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
		t += tolower(*i);
	}
	return t;
}



/************************************************************************/
/* ��Ŀ¼�������������������ļ�������                                                                      */
/************************************************************************/
static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
	printf("ɨ������Ŀ¼ %s\n", dirName.c_str());
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
		if (fileInfo.attrib&_A_SUBDIR)//�ļ�������
			continue;
		else
		{

			int i = string(fileInfo.name).find_last_of(".");
			string tempExt = toLowerCase(string(fileInfo.name).substr(i + 1));

			if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end())
			{


				fullfileName = tempPathName2 + (string(fileInfo.name));
				//  cout<<"��Ч�ļ��� '%s'\n"<< fullfileName << endl;
				fileNames.push_back((cv::String)fullfileName);



			}
			else
			{
				cout << "����ͼ���ļ�������: '%s'\n" << fileInfo.name << endl;
			}


		}

	} while (_findnext(hFile, &fileInfo) == 0);
	_findclose(hFile);
	return;

}




/************************************************************************/
/* ������������ԭѵ�����ϼ�⣬���󱨵ķ����С����Ϊ����������ԭʼ�ĸ�����
���У������ڶ���ѵ��ʹ��*/
/************************************************************************/
static long findHardExmaple(const HOGDescriptor& hog, const double hitThreshold, const vector<string>& negFileNames, const string hardExampleDir)
{

	//��Neg�м������Ķ����󱨵ģ�����߶ȼ������Ĵ�����Ϊ������
	char saveName[256];
	long hardExampleCount = 0;
	vector<Rect> foundDetection;
	// Walk over negative training samples, generate images and detect
	for (vector<string>::const_iterator negTrainingIterator = negFileNames.begin(); negTrainingIterator != negFileNames.end(); ++negTrainingIterator)
	{
		const Mat imageData = imread(*negTrainingIterator);
		hog.detectMultiScale(imageData, foundDetection, hitThreshold, winStride, Size(32,32));
		//������ͼ���м������ľ��ο򣬵õ�hard example
		for (int i = 0; i < foundDetection.size(); i++)
		{
			//�������ĺܶ���ο򶼳�����ͼ��߽磬����Щ���ο�ǿ�ƹ淶��ͼ��߽��ڲ�
			Rect r = foundDetection[i];
			if (r.x < 0)
				r.x = 0;
			if (r.y < 0)
				r.y = 0;
			if (r.x + r.width > imageData.cols)
				r.width = imageData.cols - r.x;
			if (r.y + r.height > imageData.rows)
				r.height = imageData.rows - r.y;

			//�����ο򱣴�ΪͼƬ������Hard Example
			Mat hardExampleImg = imageData(r);//��ԭͼ�Ͻ�ȡ���ο��С��ͼƬ
			resize(hardExampleImg, hardExampleImg, Size(WINSIZE_WIDTH, WINSIZE_HEIGHT));//�����ó�����ͼƬ����Ϊ64*128��С
			sprintf_s(saveName, "hardexample%09ld.jpg", hardExampleCount++);//����hard exampleͼƬ���ļ���
			imwrite(hardExampleDir + saveName, hardExampleImg);//�����ļ�
		}

	}
	printf("Hard Example Detected Done.\n");
	return hardExampleCount;


}







/************************************************************************/
/* ������                                                                     */
/************************************************************************/



int main()
{
	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9

	HOGDescriptor hog(Size(WINSIZE_WIDTH, WINSIZE_HEIGHT), Size(16, 16), Size(8,8), Size(8,8), 9);//HOG���������������HOG�����ӵ�
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm;//SVM������
	HOGDescriptor myHOG(Size(WINSIZE_WIDTH,WINSIZE_HEIGHT), Size(16, 16), Size(8, 8), Size(8, 8), 9);;

	/************************************************************************/
	/* ����ѵ�������������������������ȵ�����·��                                                                     */
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

	/*ɨ��Ŀ¼�õ�����ͼ������·��������*/
	getFilesInDirectory(testSamplesDir, testSamples, validExtensions);
	getFilesInDirectory(posTrainingSamplesDir, posTrainingSamples, validExtensions);
	getFilesInDirectory(originalNegTrainingSamplesDir, originalNegTrainingSamples, validExtensions);
	getFilesInDirectory(testNegDir, testNegSamples, validExtensions);
	getFilesInDirectory(testPosDir, testPosSamples, validExtensions);



	//Randomly generate 10 negative sample of size 64*128 for each original negative sample in /neg folder. Totally we weill ge 1,2180 neg samples

	////ͼƬ��СӦ���������ٰ���һ��64*128�Ĵ���
	if (RANDNEG_SAVED == NO)
	{
		char saveName[256];
		long CropImageCount = 0;
		for (vector<string>::iterator i = originalNegTrainingSamples.begin(); i < originalNegTrainingSamples.end(); i++)
		{
			Mat src = imread(*i);
			if (src.cols >= 128 && src.rows >= 128)
			{
				//srand(time(NULL));//�������������

				//��ÿ��ͼƬ������ü�10��64*128��С�Ĳ������˵ĸ�����
				for (int i = 0; i < 10; i++)
				{
					int x = (rand() % (src.cols - 128)); //���Ͻ�x����
					int y = (rand() % (src.rows - 128)); //���Ͻ�y����
														 //cout<<x<<","<<y<<endl;
					Mat imgROI = src(Rect(x, y, WINSIZE_WIDTH, WINSIZE_HEIGHT));
					sprintf_s(saveName, "noperson%06d.jpg", ++CropImageCount);//���ɲü����ĸ�����ͼƬ���ļ���
					imwrite(randNegSamplesDir + saveName, imgROI);//�����ļ�
				}
			}
		}
	}
	

	getFilesInDirectory(randNegSamplesDir, randNegTrainingSamples, validExtensions);



	long int totalTrainingSampleNO = posTrainingSamples.size() + randNegTrainingSamples.size();
	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��	
	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����


#if DESCRIPTOR_FIRST_SAVED == NO
	int num = 0;
	ofstream writeFeatureToFile(featuresFile.c_str(), ios::out);
	if (!(writeFeatureToFile.good() && writeFeatureToFile.is_open()))
	{
		cout << "Cannot open file to save features.\n";
	}

	cout << "����������" << endl;
	for (vector<string>::iterator iter = posTrainingSamples.begin(); iter<posTrainingSamples.end(); iter++, num++)
	{
		//out << "����" << *iter << endl;
		Mat src = imread(*iter);//��ȡͼƬ
		if (CENTRAL_CROP)
			if (src.size() == Size(96, 160))
				src = src(Rect(16, 16, 64, 128));//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������
			else
				resize(src, src, Size(64, 128));

		vector<float> descriptors;//HOG����������
		hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
												  //cout<<"������ά����"<<descriptors.size()<<endl;

												  //�洢Feature���ļ�
		writeFeatureToFile << 1 << " ";//�洢���+1��ʾ�����ˣ�-1��ʾû������
		for (vector<float>::iterator it = descriptors.begin(); it < descriptors.end(); it++)
		{
			writeFeatureToFile << *it << " ";
		}

		//�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������
		if (posTrainingSamples.begin() == iter)
		{
			DescriptorDim = descriptors.size();//HOG�����ӵ�ά��
											   //��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat

			sampleFeatureMat = Mat::zeros(totalTrainingSampleNO, DescriptorDim, CV_32FC1);
			//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
			sampleLabelMat = Mat::zeros(totalTrainingSampleNO, 1, CV_32FC1);
		}

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
		for (int i = 0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��
		sampleLabelMat.at<float>(num, 0) = 1;//���������Ϊ1������
	}

	//���ζ�ȡ������ͼƬ������HOG������
	cout << "��������" << endl;
	num = 0;
	for (vector<string>::iterator iter = randNegTrainingSamples.begin(); iter<randNegTrainingSamples.end(); iter++, num++)
	{
		cout << "����" << *iter << endl;
		//ImgName = "D:\\DataSet\\NoPersonFromINRIA\\" + ImgName;//���ϸ�������·����
		Mat src = imread(*iter);//��ȡͼƬ
		vector<float> descriptors;//HOG����������
		hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
												  //cout<<"������ά����"<<descriptors.size()<<endl;
		writeFeatureToFile << -1 << " ";//�洢���+1��ʾ�����ˣ�-1��ʾû������
		for (vector<float>::iterator it = descriptors.begin(); it < descriptors.end(); it++)
		{
			writeFeatureToFile << *it << " ";
		}

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
		for (int i = 0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(num + posTrainingSamples.size(), i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
		sampleLabelMat.at<float>(num + posTrainingSamples.size(), 0) = -1;//���������Ϊ-1������
	}


	writeFeatureToFile.close();//�ر��ļ�

							   //ѵ��SVM������ ��һ��ѵ�� û�м��� HardSamples��ѵ��
							   //������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
	vector<float> myDetector;
	getTrainedDetector(svm, criteria, param, myDetector, sampleFeatureMat, sampleLabelMat);
	saveDescriptorVectorToFile(myDetector, descriptorVectorFile);
	float temp_rho = myDetector.at(myDetector.size()-1);
	myHOG.setSVMDetector(myDetector);
	cout << "the rho is:" << temp_rho << endl;
	//���ϣ���һ��ѵ������, �õ�������



#else
	vector<float> descriptors;//HOG����������,ֻ��Ϊ�˵õ������ӵ�ά����û���κ�����;
	hog.compute(imread(randNegTrainingSamples.at(1).c_str()), descriptors, Size(8, 8));
	DescriptorDim = descriptors.size();
	sampleLabelMat = Mat::zeros(totalTrainingSampleNO, 1, CV_32FC1);
	sampleFeatureMat = Mat::zeros(totalTrainingSampleNO, descriptors.size(), CV_32FC1);

	if (DESCRIPTOR_SECOND_SAVED == NO)
	{
		loadFeatures(featuresFile, sampleFeatureMat, sampleLabelMat);

		svm.load(svmModelFile_first.c_str());
	}
	//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
	vector<float> myDetector;
	loadDescriptorFromFile(myDetector, descriptorVectorFile);
	myDetector.pop_back();//ȥ�����һ���ظ���ֵ
						  //������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
	CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
	CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
	float temp_rho = 0.0;
	myHOG.setSVMDetector(myDetector);
#endif
	cout << "��ѵ�����Ͻ������˼��(��������ο�)\n";
	detecOnSet(myHOG, 0, posTrainingSamples, originalNegTrainingSamples);

#if DESCRIPTOR_SECOND_SAVED == NO

	//�õ�hardSamples�б�
	//��������������
	cout << "Generating hard samples and save them...\n";
#if HARD_SAMPLE_SAVED == NO
	findHardExmaple(myHOG, 0, originalNegTrainingSamples, hardSampleDir);
#endif //#if HARD_SAMPLE_SAVED
	getFilesInDirectory(hardSampleDir, hardSamples, validExtensions);

	//����HardSample������  , HardSamples�Ѿ��ü���64x128��С
	//long int hardSampleNO = hardSamples.size();
	cout << "Adding  hard samples to training....\n";
	Mat hardSampleFeatureMat = Mat::zeros(hardSamples.size(), DescriptorDim, CV_32FC1);
	Mat hardSampleLabelMat = Mat::zeros(hardSamples.size(), 1, CV_32FC1);
	if (hardSamples.size() > 0)
	{
		ofstream addFeature(featuresFile.c_str(), ios::app);

		cout << "��������������\n";
		int num = 0;															 //���ζ�ȡHardExample������ͼƬ������HOG������
		for (vector<string>::iterator iter = hardSamples.begin(); iter<hardSamples.end(); iter++, num++)
		{
			/*cout << "����������" << *iter << endl;*/
			//	ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//����HardExample��������·����
			Mat src = imread(*iter);//��ȡͼƬ

			vector<float> descriptors;//HOG����������
			hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
													  //������õ�HOG�����Ӹ��Ƶ�������������hardsampleFeatureMat

													  //�洢Feature���ļ�
			addFeature << -1 << " ";//�洢���+1��ʾ�����ˣ�-1��ʾû������
			for (vector<float>::iterator it = descriptors.begin(); it < descriptors.end(); it++)
			{
				addFeature << *it << " ";
			}

			for (int i = 0; i<descriptors.size(); i++)
				hardSampleFeatureMat.at<float>(num, i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��
			hardSampleLabelMat.at<float>(num, 0) = -1;//���������Ϊ-1������
		}//end for

		addFeature.close();
	}//end if


	 //�ϲ�
	cout << "Merging the FeatureMat...\n";
	sampleFeatureMat.push_back(hardSampleFeatureMat);//�ϲ�
	sampleLabelMat.push_back(hardSampleLabelMat);



	//�ڶ���ѵ����ѵ���������ֲ��䣻
	myDetector.clear();
	getTrainedDetector(svm, criteria, param, myDetector, sampleFeatureMat, sampleLabelMat);
	//�ٴα���Descriptor
	saveDescriptorVectorToFile(myDetector, descriptorVectorFile);
	temp_rho = myDetector.at(myDetector.size()-1);
	cout << "the second time, rho is: " << temp_rho << endl;
	myHOG.setSVMDetector(myDetector);

	svm.save(svmModelFile.c_str());//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�  


								   /**************����ͼƬ����HOG���˼��******************/
								   //detectTest(myHOG, 0, testSamples);
	cout << "��ѵ�����Ͻ������˼��(��������ο�)\n";

#endif //#if DESCRIPTOR_SECOND_SAVED
	detecOnSet(myHOG, 0, posTrainingSamples,originalNegTrainingSamples);
	detectTest(myHOG, 0, testSamples);
	waitKey(1000);
	return 0;
}