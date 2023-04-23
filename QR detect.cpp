#include<iostream>
#include<opencv2\opencv.hpp>
using namespace std;
using namespace cv;

void scanAndDetectQRCode(Mat &image,int index);
bool isXCorner(Mat &image);
bool isYCorner(Mat &image);
Mat transformCorner(Mat &image, RotatedRect&rect);
int main(int argc, char** argv)
{
	Mat src = imread("D:/Image/IMAGE/opencv_tutorial_data-master/images/qrcode2.png");
	//Mat src = imread("D:/Image/IMAGE/opencv_tutorial_data-master/images/qrcode_05.jpg");
		if (src.empty())
		{
			printf("can't load image");
			return -1;
		}
		namedWindow("input", WINDOW_AUTOSIZE);
		imshow("input", src);
		scanAndDetectQRCode(src,0);
		waitKey(0);
		destroyAllWindows();
		return 0;
}
void scanAndDetectQRCode(Mat &image, int index)
{
	//颜色转换及其二值化
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("binary", binary);
	//发现轮廓
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;
	Moments monents;
	//等级制度; （教会等组织中的）掌权集团，统治集团，领导层; （观念、信仰等的）等级体系，层次体系; 级系，分级系统（如动植物的纲、目、科等）; 递阶；层级; 僧侣统治集团;
	findContours(binary.clone(), contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point());
	Mat result = Mat::zeros(image.size(), CV_8UC1);
	for (size_t t = 0; t < contours.size(); t++)
	{
		double area = contourArea(contours[t]);
		if (area < 200)continue;
		RotatedRect rect = minAreaRect(contours[t]);
		//根据矩形特征进行几何分析
		float w = rect.size.width;
		float h = rect.size.height;
		float rate = min(w, h) / max(w, h);
		if (rate > 0.85&&w < image.cols / 4 && h < image.rows / 4)
		{
			drawContours(image, contours, static_cast<int>(t), Scalar(255, 0, 0), 2, 8);
			Mat qr_roi = transformCorner(image, rect);
			if (isXCorner(qr_roi))
			{
				if (isYCorner(qr_roi))
				{
					drawContours(image, contours, static_cast<int>(t), Scalar(255, 0, 0), 2, 8);
					drawContours(result, contours, static_cast<int>(t), Scalar(255), 2, 8);
				}
			}
		}
		
	}
	imshow("binary", image);
	imshow("result", result);
}
bool isXCorner(Mat &image)
{
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imwrite("D:/Image/IMAGE/opencv_tutorial_data-master/images/change/123.png",binary);
	int xb = 0, yb = 0;
	int w1x = 0, w2x = 0;
	int b1x = 0, b2x = 0;
	int width = binary.cols;
	int height = binary.rows;
	//定义图像中心位置
	int cy = height / 2;
	int cx = width / 2;
	//定义中心位置的像素
	int pv = binary.at<uchar>(cy, cx);
	if (pv == 255)return false;
	//bool 变量  
	bool findleft = false, findright = false;
	int start = 0, end = 0;
	int offset = 0;
	//开始While循环
	while (true)
	{
		offset++;
		if ((cx - offset) <= width / 8 || (cx + offset) >= width - 1)
		{
			start = -1;
			end = -1;
			break;
		}
		pv = binary.at<uchar>(cy, cx - offset);
		if (pv == 255)
		{
			start = cx-offset;
			findleft=true;
		}
		pv = binary.at<uchar>(cy, cx + offset);
		if (pv == 255)
		{
			end = cx + offset;
			findright = true;
		}
		if (findleft&&findright)
		{
			break;
		}
	}
	if (start <= 0 || end <= 0)
	{
		return false;
	}
	xb = end - start;
	for (int col = start; col > 0; col--)
	{
		pv = binary.at<uchar>(cy, col);
		if (pv == 0)
		{
			w1x = start - col;
			break;
		}
	}
	for (int col = end; col < width - 1; col++)
	{
		pv = binary.at<uchar>(cy, col);
		if (pv == 0)
		{
			w2x = col - end;
			break;
		}
	}
	for (int col = (end+w2x); col < width; col++)
	{
		pv = binary.at<uchar>(cy, col);
		if (pv == 255)
		{
			b2x = col - end-w2x;
			break;
		}
		else 
		{
			b2x++;
		}
	}
	for (int col = (start-w1x); col >0; col--)
	{
		pv = binary.at<uchar>(cy, col);
		if (pv == 255)
		{
			b1x = start-col- w1x;
			break;
		}
		else
		{
			b1x++;
		}
	}
	float sum = xb + b1x + b2x + w1x + w2x;
	//printf("xb:%d,b1x:%d,b2x:%d,w1x:%d,w2x:%d\n",xb,b1x,b2x,w1x,w2x);
	xb = static_cast<int>((xb / sum)*7.0 + 0.5);
	b1x = static_cast<int>((b1x / sum)*7.0 + 0.5);
	b2x = static_cast<int>((b2x / sum)*7.0 + 0.5);
	w1x = static_cast<int>((w1x / sum)*7.0 + 0.5);
	w2x = static_cast<int>((w2x / sum)*7.0 + 0.5);
	//printf("xb:%d,b1x:%d,b2x:%d,w1x:%d,w2x:%d\n",xb,b1x,b2x,w1x,w2x);
	if ((xb == 3 || xb == 4) && b1x == b2x && w1x == w2x && w1x == b1x && b1x == 1)//1:1:3:1:1
	{
		return true;
	}
	else
	{
		return false;
	}
}
bool isYCorner(Mat &image)
{
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	//imwrite("D:/Image/IMAGE/opencv_tutorial_data-master/images/change/123.png", binary);
	int width = binary.cols;
	int height = binary.rows;
	int cy = height / 2;
	int cx = width / 2;
	int pv = binary.at<uchar>(cy, cx);
	int bc = 0, wc = 0;
	bool found = true;
	for (int row = cy; row > 0; row--)
	{
		pv = binary.at<uchar>(row, cx);
		if (pv == 0 && found)
		{
			bc++;
		}
		else if (pv == 255)
		{
			found = false;
			wc++;
		}
		bc = bc * 2;
		if (bc <= wc)
		{
			return false;
		}
		return true;
	}
}
Mat transformCorner(Mat &image, RotatedRect&rect)
{
//透视变换
	int width = static_cast<int>(rect.size.width);
	int height = static_cast<int>(rect.size.height);
	Mat result = Mat::zeros(height, width, image.type());
	Point2f vertices[4];
	rect.points(vertices);
	vector<Point>src_corners;
	vector<Point>dst_corners;
	dst_corners.push_back(Point(0, 0));
	dst_corners.push_back(Point(width, 0));
	dst_corners.push_back(Point(width, height));
	dst_corners.push_back(Point(0, height));
	for (int i = 0; i < 4; i++)
	{
		src_corners.push_back(vertices[i]);
	}
	Mat h = findHomography(src_corners, dst_corners);
	/*
	findHomography： 透视变换，计算多个二维点对之间的最优单映射变换矩阵 H（3行x3列） ，使用最小均方误差或者RANSAC方法
	函数功能：找到两个平面之间的转换矩阵。
      Mat cv::findHomography	(	InputArray 	srcPoints,
								InputArray 	dstPoints,
								int 	method = 0,
								double 	ransacReprojThreshold = 3,
								OutputArray 	mask = noArray(),
								const int 	maxIters = 2000,
								const double 	confidence = 0.995
)
	*/
	warpPerspective(image, result, h, result.size());
	/*C++: void warpPerspective(InputArray src, OutputArray dst, InputArray M, Size dsize, 
	int flags=INTER_LINEAR, int borderMode=BORDER_CONSTANT, const Scalar& borderValue=Scalar())
参数详解：
InputArray src：输入的图像
OutputArray dst：输出的图像
InputArray M：透视变换的矩阵
Size dsize：输出图像的大小
int flags=INTER_LINEAR：输出图像的插值方法，
combination of interpolation methods (INTER_LINEAR or INTER_NEAREST) and the optional 
     flagWARP_INVERSE_MAP, that sets M as the inverse transformation (  )
int borderMode=BORDER_CONSTANT：图像边界的处理方式
const Scalar& borderValue=Scalar()：边界的颜色设置，一般默认是0
*/
	return result;

}