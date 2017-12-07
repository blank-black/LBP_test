#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#define  PI 3.1415926
#define BASE 2.0

typedef struct MyPoint
{
	double x;
	double y;
}MyPoint;

void init_image_data(int height, int width, int num_sp, MyPoint *spoint, int d[],int orign[])
{
	int max_x = 0, max_y = 0, min_x = 0, min_y = 0;
	for (int k = 0; k < num_sp; k++)
	{
		if (max_x < spoint[k].x)
			max_x = spoint[k].x;
		if (max_y < spoint[k].y)
			max_y = spoint[k].y;
		if (min_x > spoint[k].x)
			min_x = spoint[k].x;
		if (min_y > spoint[k].y)
			min_y = spoint[k].y;
	}

	//计算模版大小
	int box_x = ceil(MAX(max_x, 0)) - floor(MIN(min_x, 0)) + 1;
	int box_y = ceil(MAX(max_y, 0)) - floor(MIN(min_y, 0)) + 1;

	if (width < box_x || height < box_y)
	{
		printf("Too small input image. Should be at least (2*radius+1) x (2*radius+1)");
		return;
	}

	//计算可滤波图像大小,opencv图像数组下标从0开始
	orign[0] = 0 - floor(MIN(min_x, 0));//起点
	orign[1] = 0 - floor(MIN(min_x, 0));

	d[0] = width - box_x + 1;//终点
	d[1] = height - box_y + 1;
}

//	target = cvCreateImage(cvSize(d[0], d[1]), IPL_DEPTH_8U, 1);
//	result = (int *) malloc(sizeof(int) * d[0] * d[1]);



void linear_interpolation(IplImage *src, int num_sp, MyPoint *spoint, int d[],int orign[],
                          IplImage *target,int result[], int t[],int f[], int c[],double w[])
{
	memset(result, 0, sizeof(result));
	CvRect roi = cvRect(orign[0], orign[1], d[0], d[1]);
	cvSetImageROI(src, roi);
	cvCopy(src, target);
	cvResetImageROI(src);
	cvSaveImage("/Users/zby/Documents/GitHub/CLion_test/test_LBP/pic_result/pic.jpg", src);

	for (int k = 0; k < num_sp; k++)
	{
		double x = spoint[k].x + orign[0];
		double y = spoint[k].y + orign[1];

		//二线性插值图像
		f[1] = floor(y);  //向下取整
		f[0] = floor(x);
		c[1] = ceil(y);   //向上取整
		c[0] = ceil(x);
		t[1] = y - f[1];
		t[0] = x - f[0];
		w[0] = (1 - t[0]) * (1 - t[1]);
		w[1] = t[0] * (1 - t[1]);
		w[2] = (1 - t[0]) * t[1];
		w[3] = t[0] * t[1];
		int v = pow(BASE, (float) k);

		for (int i = 0; i < d[1]; i++)
			for (int j = 0; j < d[0]; j++)
			{
				//灰度插值图像像素
				double N = w[0] * (double) (unsigned char) src->imageData[(i + f[1]) * src->width + j + f[0]] +
				           w[1] * (double) (unsigned char) src->imageData[(i + f[1]) * src->width + j + c[0]] +
				           w[2] * (double) (unsigned char) src->imageData[(i + c[1]) * src->width + j + f[0]] +
				           w[3] * (double) (unsigned char) src->imageData[(i + c[1]) * src->width + j + c[0]];

				if (N >= (double) (unsigned char) target->imageData[i * d[0] + j])
					result[i * d[0] + j] = result[i * d[0] + j] + v * 1;
				else
					result[i * d[0] + j] = result[i * d[0] + j] + v * 0;
			}
	}
}





/////////////////////
///   灰度不变性
/////////////////////

void gray_invariant_lbp(IplImage *src, int height, int width, int num_sp, MyPoint *spoint)
{
	IplImage *target, *hist;
	int i, j, orign[2], d[2], t[2], f[2], c[2];
	double w[4];
	int *result;
	init_image_data(height, width, num_sp, spoint, d, orign);

	target = cvCreateImage(cvSize(d[0], d[1]), IPL_DEPTH_8U, 1);
	result = (int *) malloc(sizeof(int) * d[0] * d[1]);

	linear_interpolation(src, num_sp, spoint, d, orign, target, result, t, f, c, w);

	int cols = pow(BASE, (float) num_sp);
	hist = cvCreateImage(cvSize(300, 200), IPL_DEPTH_8U, 3);//直方图图像
	double *val_hist = (double *) malloc(sizeof(double) * cols);   //直方图数组

	//显示图像
	if (num_sp <= 8)
	{
		//只有采样数小于8，则编码范围0-255，才能显示图像
		for (i = 0; i < d[1]; i++)
			for (j = 0; j < d[0]; j++)
				target->imageData[i * d[0] + j] = (unsigned char) result[i * d[0] + j];
		cvSaveImage("/Users/zby/Documents/GitHub/CLion_test/test_LBP/pic_result/gray_result.jpg", target);
	}

	//显示直方图

	for (i = 0; i < cols; i++)
		val_hist[i] = 0.0;
	for (i = 0; i < d[1] * d[0]; i++)
		val_hist[result[i]]++;

	double temp_max = 0.0;

	for (i = 0; i < cols; i++)         //求直方图最大值，为了归一化
	{
		//printf("%f\n",val_hist[i]);
		if (temp_max < val_hist[i])
			temp_max = val_hist[i];
	}
	//画直方图
	CvPoint p1, p2;
	double bin_width = (double) hist->width / cols;
	double bin_height = (double) hist->height / temp_max;

	for (i = 0; i < cols; i++)
	{
		p1.x = i * bin_width;
		p1.y = hist->height;
		p2.x = (i + 1) * bin_width;
		p2.y = hist->height - val_hist[i] * bin_height;
		cvRectangle(hist, p1, p2, cvScalar(0, 255), -1, 8, 0);
	}
	cvSaveImage("/Users/zby/Documents/GitHub/CLion_test/test_LBP/pic_result/gray_histogram.jpg", hist);
}





/////////////////////
///    旋转不变性
/////////////////////
void rotation_invariant_mapping(int range,int num_sp,int *Mapping)
{
	int newMax, rm, r;
	int *tmpMap;

	newMax = 0;
	tmpMap = (int *) malloc(sizeof(int) * range);
	memset(tmpMap, -1, sizeof(int) * range);

	for (int i = 0; i < range; i++)
	{
		rm = i;
		r = i;
		for (int j = 0; j < num_sp - 1; j++)
		{
			//将r向左循环移动一位,当r超过num_sp位时，舍弃
			r = r << 1;
			if (r > range - 1)
				r = r - (range - 1);
			//printf("%d,%d\n",r,rm);
			if (r < rm)
				rm = r;
		}
		if (tmpMap[rm] < 0)
		{
			tmpMap[rm] = newMax;
			newMax++;
		}
		Mapping[i] = tmpMap[rm];
	}
	free(tmpMap);
}

void rotation_invariant_lbp(IplImage *src,int height,int width,int num_sp,MyPoint *spoint,int *Mapping)
{
	IplImage *target, *hist;
	int i, j, orign[2], d[2], t[2], f[2], c[2];
	double w[4];
	int *result;

	init_image_data(height, width, num_sp, spoint, d, orign);

	target = cvCreateImage(cvSize(d[0], d[1]), IPL_DEPTH_8U, 1);
	result = (int *) malloc(sizeof(int) * d[0] * d[1]);

	linear_interpolation(src, num_sp, spoint, d, orign, target, result, t, f, c, w);

	//将result的值映射为mapping的值
	for (i = 0; i < d[1]; i++)
		for (j = 0; j < d[0]; j++)
			result[i * d[0] + j] = Mapping[result[i * d[0] + j]];

	//显示图像
	int cols = 0;//直方图的横坐标，也是result数组的元素种类
	int mapping_size = pow(BASE, (float) num_sp);
	for (i = 0; i < mapping_size; i++)
		if (cols < Mapping[i])
			cols = Mapping[i];

	if (cols < 255)
	{
		//只有采样数小于8，则编码范围0-255，才能显示图像
		for (i = 0; i < d[1]; i++)
			for (j = 0; j < d[0]; j++)
			{
				target->imageData[i * d[0] + j] = (unsigned char) result[i * d[0] + j];
				//printf("%d\n",(unsigned char)target->imageData[i*width+j]);
			}
		cvSaveImage("/Users/zby/Documents/GitHub/CLion_test/test_LBP/pic_result/rotation_result.jpg", target);
	}

	//计算直方图
	hist = cvCreateImage(cvSize(300, 200), IPL_DEPTH_8U, 3);//直方图图像

	double *val_hist = (double *) malloc(sizeof(double) * cols);   //直方图数组
	for (i = 0; i < cols; i++)
		val_hist[i] = 0.0;
	for (i = 0; i < d[1] * d[0]; i++)
		val_hist[result[i]]++;

	double temp_max = 0.0;

	for (i = 0; i < cols; i++)         //求直方图最大值，为了归一化
	{
		//printf("%f\n",val_hist[i]);
		if (temp_max < val_hist[i])
			temp_max = val_hist[i];
	}
	//画直方图
	CvPoint p1, p2;
	double bin_width = (double) hist->width / cols;
	double bin_height = (double) hist->height / temp_max;

	for (i = 0; i < cols; i++)
	{
		p1.x = i * bin_width;
		p1.y = hist->height;
		p2.x = (i + 1) * bin_width;
		p2.y = hist->height - val_hist[i] * bin_height;
		cvRectangle(hist, p1, p2, cvScalar(0, 255), -1, 8, 0);
	}
	cvSaveImage("/Users/zby/Documents/GitHub/CLion_test/test_LBP/pic_result/rotation_histogram.jpg", hist);
}






/////////////////////
///uniform 旋转不变性
/////////////////////
void calc_position(int radius,int num_sp,MyPoint *spoint)
{
	double theta;

	theta = 2 * PI / num_sp;

	for (int i = 0; i < num_sp; i++)
	{
		spoint[i].y = -radius * sin(i * theta);
		spoint[i].x = radius * cos(i * theta);
	}
}
int calc_sum(int r)
{
	int res_sum;

	res_sum = 0;
	while (r)
	{
		res_sum = res_sum + r % 2;
		r /= 2;
	}
	return res_sum;
}
void rotation_uniform_invariant_mapping(int range,int num_sp,int *Mapping)
{
	int numt, i, j, tem_xor;

	numt = 0;
	tem_xor = 0;
	for (i = 0; i < range; i++)
	{
		j = i << 1;
		if (j > range - 1)
			j = j - (range - 1);

		tem_xor = i ^ j;    // 异或
		numt = calc_sum(tem_xor);//计算异或结果中1的个数，即跳变个数

		if (numt <= 2)
			Mapping[i] = calc_sum(i);
		else
			Mapping[i] = num_sp + 1;
	}

}

void rotation_uniform_invariant_lbp(IplImage *src,int height,int width,int num_sp,MyPoint *spoint,int *Mapping)
{
	IplImage *target, *hist;
	int i, j, orign[2], d[2], t[2], f[2], c[2];
	double w[4];
	int *result;

	init_image_data(height, width, num_sp, spoint, d, orign);

	target = cvCreateImage(cvSize(d[0], d[1]), IPL_DEPTH_8U, 1);
	result = (int *) malloc(sizeof(int) * d[0] * d[1]);

	linear_interpolation(src, num_sp, spoint, d, orign, target, result, t, f, c, w);

	//将result的值映射为mapping的值
	for (i = 0; i < d[1]; i++)
		for (j = 0; j < d[0]; j++)
			result[i * d[0] + j] = Mapping[result[i * d[0] + j]];

	//显示图像
	int cols = 0;//直方图的横坐标，也是result数组的元素种类
	int mapping_size = pow(BASE, (float) num_sp);
	for (i = 0; i < mapping_size; i++)
		if (cols < Mapping[i])
			cols = Mapping[i];

	if (cols < 255)
	{
		//只有采样数小于8，则编码范围0-255，才能显示图像
		for (i = 0; i < d[1]; i++)
			for (j = 0; j < d[0]; j++)
			{
				target->imageData[i * d[0] + j] = (unsigned char) result[i * d[0] + j];
				//printf("%d\n",(unsigned char)target->imageData[i*width+j]);
			}
		cvSaveImage("/Users/zby/Documents/GitHub/CLion_test/test_LBP/pic_result/rotation_uniform_result.jpg", target);
	}

	//计算直方图
	hist = cvCreateImage(cvSize(300, 200), IPL_DEPTH_8U, 3);//直方图图像
	double *val_hist = (double *) malloc(sizeof(double) * cols);   //直方图数组
	for (i = 0; i < cols; i++)
		val_hist[i] = 0.0;
	for (i = 0; i < d[1] * d[0]; i++)
		val_hist[result[i]]++;

	double temp_max = 0.0;

	for (i = 0; i < cols; i++)         //求直方图最大值，为了归一化
	{
		//printf("%f\n",val_hist[i]);
		if (temp_max < val_hist[i])
			temp_max = val_hist[i];
	}
	//画直方图
	CvPoint p1, p2;
	double bin_width = (double) hist->width / cols;
	double bin_height = (double) hist->height / temp_max;

	for (i = 0; i < cols; i++)
	{
		p1.x = i * bin_width;
		p1.y = hist->height;
		p2.x = (i + 1) * bin_width;
		p2.y = hist->height - val_hist[i] * bin_height;
		cvRectangle(hist, p1, p2, cvScalar(0, 255), -1, 8, 0);
	}
	cvSaveImage("/Users/zby/Documents/GitHub/CLion_test/test_LBP/pic_result/rotation_uniform_histogram.jpg", hist);
}

int main()
{
	IplImage *src, *grey;
	int samples, radius, range, *mapping;
	MyPoint *spoint;
	float Mi;

	samples = 8;
	radius = 10;
	Mi = 2.0;
	range = pow(Mi, samples);

	src = cvLoadImage("/Users/zby/Documents/GitHub/CLion_test/test_LBP/tjlogo.jpg");
	grey = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);
	cvCvtColor(src, grey, CV_BGR2GRAY);
	mapping = (int *) malloc(sizeof(int) * range);
	memset(mapping, 0, sizeof(int) * range);

	//计算采样点相对坐标
	spoint = (MyPoint *) malloc(sizeof(MyPoint) * samples);
	calc_position(radius, samples, spoint);

	//计算灰度不变性LBP特征，写回浮点数图像矩阵中
	gray_invariant_lbp(grey, src->height, src->width, samples, spoint);

	//计算旋转不变形LBP特征
	rotation_invariant_mapping(range, samples, mapping);
	rotation_invariant_lbp(grey, src->height, src->width, samples, spoint, mapping);

	//计算旋转不变等价LBP特征
	rotation_uniform_invariant_mapping(range, samples, mapping);
	rotation_uniform_invariant_lbp(grey, src->height, src->width, samples, spoint, mapping);
	return 0;
}



