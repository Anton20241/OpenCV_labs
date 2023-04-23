#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core.hpp>
#include <cmath>
#include <complex>
#include <time.h>
#include "cvDirectory.h"

using namespace cv;

enum dftType{
    DIRECT_DFT,
    INVERSE_DFT
};

enum cropType{
    LOW_FREQ,
    HIGH_FREQ
};

void matPrint(const std::string name, const Mat& mat);
void get_W_matrix(Mat& W_matrix, const int size, size_t type);
void one_dimensional_dft(const Mat& W_matrix, Mat& src, int i,  size_t type);
void user_dft(Mat& src_float, Mat& dst);
void fast_user_dft(Mat& src_float, Mat& dst);
void take_OpenCV_DFT(Mat& src, Mat& dst);
void krasivSpektr(Mat &magI);
void showDFT(const Mat& src, std::string name);
void convolution(Mat src, Mat filter, std::string srcName, std::string filterName, std::string resultName);
void user_invert_dft(Mat& dft_img, Mat& imgFromDft);
void dft_invDFT_Part(Mat& input_img, Mat& dft_img);
void convolution_Part(Mat& src);
void crop_Part(Mat& src);
void cropFreq(Mat& src, Mat& dst, size_t type);
void car_numbers_Part();
void findNumber(const cv::Mat& src, const cv::Mat& templ, std::string name);


int main() {

  /////////////////////////////////////////////////////////////////////////////////////
  std::string addrImages = "./";
  std::vector<std::string> imagesData = cv::Directory::GetListFiles(addrImages);
  assert(!imagesData.empty());

  Mat input_img;
  input_img = cv::imread(addrImages + imagesData[1], cv::IMREAD_GRAYSCALE);
  resize(input_img, input_img, cv::Size(), 0.4, 0.4);
  assert(!input_img.empty());
  //imshow("input_img", input_img);
  /////////////////////////////////////////////////////////////////////////////////////

  //// dft_invDFT_Part part
  //Mat dft_first_part_img;
  //dft_invDFT_Part(input_img, dft_first_part_img);

  //// convolution_Part
  //convolution_Part(input_img);

  //// convolution_Part
  //crop_Part(input_img);

  //// convolution_Part
  car_numbers_Part();

  return 0;
}

void car_numbers_Part(){
  Mat car_number = imread("car_number.jpg",cv::IMREAD_GRAYSCALE);
  assert(!car_number.empty());
  imshow("car_number", car_number);

  Mat symbol_0 = imread("0.jpg",cv::IMREAD_GRAYSCALE);
  assert(!symbol_0.empty());

  Mat symbol_7 = imread("7.jpg",cv::IMREAD_GRAYSCALE);
  assert(!symbol_7.empty());

  Mat symbol_A = imread("A.jpg",cv::IMREAD_GRAYSCALE);
  assert(!symbol_A.empty());

  findNumber(car_number, symbol_0, "0");
  findNumber(car_number, symbol_7, "7");
  findNumber(car_number, symbol_A, "A");
}

void findNumber(const cv::Mat& src, const cv::Mat& templ, std::string name){
  Size src_dftSize;
  src_dftSize.width = getOptimalDFTSize(src.cols + templ.cols - 1);
  src_dftSize.height = getOptimalDFTSize(src.rows + templ.rows - 1);

  // dft src
  Mat src_float;
  src.convertTo(src_float, CV_32FC1, 1.0/ 255.0);
  Mat srcExt(src_dftSize, CV_32FC1, Scalar(0));
  Mat tempROI1(srcExt, Rect(0, 0, src_float.cols, src_float.rows));
  src_float.copyTo(tempROI1);
  Mat srcDFT;
  dft(srcExt, srcDFT, DFT_COMPLEX_OUTPUT);

  // dft filter
  Mat templ_float;
  templ.convertTo(templ_float, CV_32FC1, 1.0/ 255.0);
  Mat filterExt(src_dftSize, CV_32FC1, Scalar(0));
  Mat tempROI2(filterExt, Rect(0, 0, templ_float.cols, templ_float.rows));
  templ_float.copyTo(tempROI2);
  Mat templDFT;
  dft(filterExt, templDFT, DFT_COMPLEX_OUTPUT);

  // spectrums multiply
  Mat multipliedSpec;
  mulSpectrums(srcDFT, templDFT, multipliedSpec, 0, 1);

  // inverse DFT to result
  Mat convRes;
  dft(multipliedSpec, convRes, DFT_INVERSE | DFT_REAL_OUTPUT);
  Mat result(src_float.rows, src_float.cols, CV_32FC1, Scalar(0));
  Mat resultROI(convRes, Rect(0, 0, src_float.cols, src_float.rows));
  resultROI.copyTo(result);
  normalize(result, result, 0, 1, NormTypes::NORM_MINMAX);

  double max_value;
  cv::minMaxLoc(result, nullptr, &max_value);
  cv::Mat img_bin;
  cv::threshold(result, img_bin, max_value - 0.01, 1, cv::THRESH_BINARY);

  // show results
//  showDFT(srcDFT, name + "_srcDFT");
  showDFT(templDFT, name + "_templDFT");
  showDFT(multipliedSpec,name + "_multipliedSpec");
  imshow(name + "_img_bin", img_bin);
  cv::waitKey();
}

void crop_Part(Mat& src){
  Size src_dftSize;
  src_dftSize.width = getOptimalDFTSize(src.cols);
  src_dftSize.height = getOptimalDFTSize(src.rows);

  // dft src
  Mat src_float;
  src.convertTo(src_float, CV_32FC1, 1.0/ 255.0);
  Mat srcExt(src_dftSize, CV_32FC1, Scalar(0));
  Mat tempROI1(srcExt, Rect(0, 0, src_float.cols, src_float.rows));
  src_float.copyTo(tempROI1);
  Mat srcDFT;
  dft(srcExt, srcDFT, DFT_COMPLEX_OUTPUT);
//  showDFT(srcDFT, "srcDFT");
//  cv::waitKey();

  Mat srcDFTLowFreq;
  Mat srcDFTHighFreq;
  srcDFT.copyTo(srcDFTLowFreq);
  srcDFT.copyTo(srcDFTHighFreq);

  Mat resLowFreq;
  Mat resHighFreq;
  src.copyTo(resLowFreq);
  src.copyTo(resHighFreq);
  cropFreq(srcDFTLowFreq, resLowFreq, LOW_FREQ);
  cropFreq(srcDFTHighFreq, resHighFreq, HIGH_FREQ);

  imshow("resLowFreq", resLowFreq);
  imshow("resHighFreq", resHighFreq);
  cv::waitKey();
}

void cropFreq(Mat& srcDFT, Mat& dst, size_t type){
  krasivSpektr(srcDFT);
  if(type == HIGH_FREQ){
    cv::circle(srcDFT, cv::Point(srcDFT.cols / 2, srcDFT.rows / 2), 30, cv::Scalar::all(0), -1);
  } else {
    Mat mask = srcDFT.clone();
    circle(mask, Point(srcDFT.cols / 2, srcDFT.rows / 2), 30, Scalar::all(0), -1);
    cv::bitwise_xor(srcDFT, mask, srcDFT);
  }

  showDFT(srcDFT, std::to_string(type) + "_srcDFT");
  cv::waitKey();

  krasivSpektr(srcDFT);

  // inverse DFT to result
  Mat convRes;
  dft(srcDFT, convRes, DFT_INVERSE | DFT_REAL_OUTPUT);
  Mat result(dst.rows, dst.cols, CV_32FC1, Scalar(0));
  Mat resultROI(convRes, Rect(0, 0, dst.cols, dst.rows));
  resultROI.copyTo(result);
  normalize(result, result, 0, 255, NormTypes::NORM_MINMAX);
  result.convertTo(result, CV_8UC1);
  result.copyTo(dst);
}

void matPrint(const std::string name, const Mat& mat){
  std::cout << "\n[" << name << " x" << "]:\n";
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      printf("[%.1f]\t", i, j, mat.at<Vec2f>(i,j)[0]);
    }
    std::cout << std::endl;
  }

  std::cout << "\n[" << name << " y" << "]:\n";
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      printf("[%.1f]\t", i, j, mat.at<Vec2f>(i,j)[1]);
    }
    std::cout << std::endl;
  }
}

void get_W_matrix(Mat& W_matrix, const int size, size_t type){

  Mat originalComplex[2] = {Mat::zeros(size, size, CV_32F), Mat::zeros(size, size, CV_32F)};
  merge(originalComplex, 2, W_matrix);
  std::complex<float> W_const_complex;

  if (type == DIRECT_DFT){
    W_const_complex.real(cos(-2 * M_PI / size));
    W_const_complex.imag(sin(-2 * M_PI / size));
  }

  if (type == INVERSE_DFT){
    W_const_complex.real(cos(2 * M_PI / size));
    W_const_complex.imag(sin(2 * M_PI / size));
  }

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      W_matrix.at<Vec2f>(i,j)[0] = std::pow( W_const_complex, i * j ).real();
      W_matrix.at<Vec2f>(i,j)[1] = std::pow( W_const_complex, i * j ).imag();
    }
  }
}

void one_dimensional_dft(const Mat& W_matrix, Mat& src, int i,  size_t type){

  Mat tmp_dst;
  src.copyTo(tmp_dst);
  assert(W_matrix.cols == tmp_dst.cols);

  int col;
  for (int row = 0; row < W_matrix.rows; row++) {
    float xx_temp = 0;
    float yy_temp = 0;
    for (col = 0; col < W_matrix.cols; col++) {

      float a = W_matrix.at<Vec2f>(row, col)[0];
      float b = W_matrix.at<Vec2f>(row, col)[1];
      float x = src.at<Vec2f>(i, col)[0];
      float y = src.at<Vec2f>(i, col)[1];
      float x_temp = 0;
      float y_temp = 0;

      if (type == DIRECT_DFT){
        x_temp = (a * x - b * y);
        y_temp = (a * y + b * x);
      }

      if (type == INVERSE_DFT){
        x_temp = (a * x - b * y) / W_matrix.rows;
        y_temp = (a * y + b * x) / W_matrix.rows;
      }

      xx_temp += x_temp;
      yy_temp += y_temp;

      tmp_dst.at<Vec2f>(i,row)[0] = xx_temp;
      tmp_dst.at<Vec2f>(i,row)[1] = yy_temp;
    }


//      printf("\n[i = %i, col = %i], %f\n", i, row, tmp_dst.at<Vec2f>(i,row)[0]);
//      printf("\n[i = %i, col = %i], %f\n", i, row, tmp_dst.at<Vec2f>(i,row)[1]);
  }
  tmp_dst.copyTo(src);
}

void user_dft(Mat& src_float, Mat& dst){ //src_float (0...1) - 1 channel, dst - 2 channel

  //get W_matrix_X
  Mat W_matrix_X;
  get_W_matrix(W_matrix_X, src_float.cols, DIRECT_DFT);
  //matPrint("W_matrix_X", W_matrix_X);

  //get W_matrix_Y
  Mat W_matrix_Y;
  get_W_matrix(W_matrix_Y, src_float.rows, DIRECT_DFT);
  //matPrint("W_matrix_Y", W_matrix_Y);

  Mat originalComplex[2] = {src_float, Mat::zeros(src_float.size(), CV_32F)};
  merge(originalComplex, 2, dst);

  //get DFT for every row
  for (int i = 0; i < dst.rows; ++i) {
    one_dimensional_dft(W_matrix_X, dst, i, DIRECT_DFT);
  }

  //matPrint("dst", dst);

  dst = dst.t();
  //matPrint("dstTranspose", dst);

  //get DFT for every row
  for (int i = 0; i < dst.rows; ++i) {
    one_dimensional_dft(W_matrix_Y, dst, i, DIRECT_DFT);
  }

  dst = dst.t();

  //matPrint("dst", dst);
}

void fast_user_dft(Mat& src_float, Mat& dst){ //src_float (0...1) - 1 channel, dst - 2 channel

}

void take_USER_DFT(Mat& src, Mat& dst){
  Mat src_float;
  src.convertTo(src_float, CV_32FC1, 1.0/ 255.0);
  //matPrint("src_float", src_float);
  user_dft(src_float, dst);
}

void take_FAST_USER_DFT(Mat& src, Mat& dst){
  Mat src_float;
  src.convertTo(src_float, CV_32FC1, 1.0/ 255.0);
  //matPrint("src_float", src_float);
  fast_user_dft(src_float, dst);
}

void take_OpenCV_DFT(Mat& src, Mat& dst){
  Mat src_float;
  src.convertTo(src_float, CV_32FC1, 1.0/ 255.0);
//  matPrint("src_float", src_float);
//  Mat originalComplex[2] = {src_float, Mat::zeros(src_float.size(), CV_32F)};
//  merge(originalComplex, 2, dst);
  dft(src_float, dst, cv::DFT_COMPLEX_OUTPUT);
}

void krasivSpektr(Mat &magI){
  // rearrange the quadrants of Fourier image  so that the origin is at the image center
  int cx = magI.cols / 2;
  int cy = magI.rows / 2;

  Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
  Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
  Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
  Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

  Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);
}

void showDFT(const Mat& src, std::string name){
  Mat splitMat[2] = {Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F)};
  split(src, splitMat);
  Mat dftMagnitude;
  magnitude(splitMat[0], splitMat[1], dftMagnitude);
  //krasivSpektr(dftMagnitude);

  dftMagnitude += Scalar::all(1);
  log(dftMagnitude, dftMagnitude);
  normalize(dftMagnitude, dftMagnitude, 0, 1, NormTypes::NORM_MINMAX);

  Mat spectrum_show;
  dftMagnitude.convertTo(spectrum_show, CV_8UC1, 255);
  imshow(name, spectrum_show);
}

void convolution(Mat src, Mat filter, std::string srcName, std::string filterName, std::string resultName){

  Size src_dftSize;
  src_dftSize.width = getOptimalDFTSize(src.cols + filter.cols - 1);
  src_dftSize.height = getOptimalDFTSize(src.rows + filter.rows - 1);

  // dft src
  Mat src_float;
  src.convertTo(src_float, CV_32FC1, 1.0/ 255.0);
  Mat srcExt(src_dftSize, CV_32FC1, Scalar(0));
  Mat tempROI1(srcExt, Rect(0, 0, src_float.cols, src_float.rows));
  src_float.copyTo(tempROI1);
  Mat srcDFT;
  dft(srcExt, srcDFT, DFT_COMPLEX_OUTPUT);

  // dft filter
  Mat filter_float;
  filter.convertTo(filter_float, CV_32FC1, 1.0/ 255.0);
  Mat filterExt(src_dftSize, CV_32FC1, Scalar(0));
  Mat tempROI2(filterExt, Rect(0, 0, filter_float.cols, filter_float.rows));
  filter_float.copyTo(tempROI2);
  Mat filterDFT;
  dft(filterExt, filterDFT, DFT_COMPLEX_OUTPUT);

  // spectrums multiply
  Mat multipliedSpec;
  mulSpectrums(srcDFT, filterDFT, multipliedSpec, 0, false);

  // inverse DFT to result
  Mat convRes;
  dft(multipliedSpec, convRes, DFT_INVERSE | DFT_REAL_OUTPUT);
  Mat result(src_float.rows, src_float.cols, CV_32FC1, Scalar(0));
  Mat resultROI(convRes, Rect(0, 0, src_float.cols, src_float.rows));
  resultROI.copyTo(result);
  normalize(result, result, 0, 255, NormTypes::NORM_MINMAX);
  result.convertTo(result, CV_8UC1);

  // show results
  showDFT(srcDFT, srcName);
  showDFT(filterDFT, filterName);
  showDFT(multipliedSpec, filterName + "multipliedSpec");
  imshow(resultName, result);
  cv::waitKey();
}

void user_invert_dft(Mat& dft_img, Mat& imgFromDft){

  //get W_matrix_X
  Mat W_matrix_X;
  get_W_matrix(W_matrix_X, dft_img.cols, INVERSE_DFT);
  //matPrint("W_matrix_X", W_matrix_X);

  //get W_matrix_Y
  Mat W_matrix_Y;
  get_W_matrix(W_matrix_Y, dft_img.rows, INVERSE_DFT);
  //matPrint("W_matrix_Y", W_matrix_Y);

  dft_img.copyTo(imgFromDft);

  //get DFT for every row
  for (int i = 0; i < imgFromDft.rows; ++i) {
    one_dimensional_dft(W_matrix_X, imgFromDft, i, INVERSE_DFT);
  }

  //matPrint("dst", dst);

  imgFromDft = imgFromDft.t();
  //matPrint("dstTranspose", dst);

  //get DFT for every row
  for (int i = 0; i < imgFromDft.rows; ++i) {
    one_dimensional_dft(W_matrix_Y, imgFromDft, i, INVERSE_DFT);
  }

  imgFromDft = imgFromDft.t();
  //matPrint("dst", dst);

  Mat splitChannels[2];
  split(imgFromDft, splitChannels);
  imgFromDft = splitChannels[0];
}

void dft_invDFT_Part(Mat& input_img, Mat& dft_img){
  clock_t begin;
  clock_t end;

  begin = clock();
  take_USER_DFT(input_img, dft_img);
  end = clock();
  std::cout << "take_USER_DFT: " << (double)(end - begin) / CLOCKS_PER_SEC << std::endl;
  showDFT(dft_img, "USER_DFT");


  //take_FAST_USER_DFT(input_img, dft_img);

  begin = clock();
  take_OpenCV_DFT(input_img, dft_img);
  end = clock();
  std::cout << "take_OpenCV_DFT: " << (double)(end - begin) / CLOCKS_PER_SEC << std::endl;

  Mat imgFromDft;
//
//  begin = clock();
//  dft(dft_img, imgFromDft, DFT_INVERSE|DFT_REAL_OUTPUT|DFT_SCALE);
//  end = clock();
//  std::cout << "take_OpenCV_invert_DFT: " << (double)(end - begin) / CLOCKS_PER_SEC << std::endl;

//  begin = clock();
//  user_invert_dft(dft_img, imgFromDft);
//  end = clock();
//  std::cout << "user_invert_dft: " << (double)(end - begin) / CLOCKS_PER_SEC << std::endl;

//  imshow("imgFromDft", imgFromDft);
//  showDFT(dft_img, "DFT");

  cv::waitKey();
}

void convolution_Part(Mat& src){
  Mat box = (Mat_<double>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
  Mat horSobel = (Mat_<double>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
  Mat verSobel = (Mat_<double>(3, 3) << 1, 0, -1, 2, 0,-2, 1, 0,-1);
  Mat laplace = (Mat_<double>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);

  convolution(src, box, "dft src", "dft box filter", "box filter result");
  convolution(src, horSobel, "dft src", "dft horSobel filter", "horSobel filter result");
  convolution(src, verSobel, "dft src", "dft verSobel filter", "verSobel filter result");
  convolution(src, laplace, "dft src", "dft laplace filter", "laplace filter result");
}