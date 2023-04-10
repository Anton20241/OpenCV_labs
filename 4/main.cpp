#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core.hpp>
#include <cmath>
#include <complex>

using namespace cv;

enum dftType{
    DIRECT_DFT,
    INVERSE_DFT
};

void matPrint(const std::string name, const Mat& mat){
  std::cout << "\n[" << name << " x" << "]:\n";
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      printf("[%0.1f]\t", i, j, mat.at<Vec2f>(i,j)[0]);
    }
    std::cout << std::endl;
  }

  std::cout << "\n[" << name << " y" << "]:\n";
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      printf("[%0.1f]\t", i, j, mat.at<Vec2f>(i,j)[1]);
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
  merge(originalComplex, 2, src_float);

  src_float.copyTo(dst);

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

void take_USER_DFT(Mat& src, Mat& dst){
  Mat src_float;
  src.convertTo(src_float, CV_32FC1, 1.0/ 255.0);
  //matPrint("src_float", src_float);
  user_dft(src_float, dst);
}

void take_OpenCV_DFT(Mat& src, Mat& dst){
  Mat src_float;
  src.convertTo(src_float, CV_32FC1, 1.0/ 255.0);
  //matPrint("src_float", src_float);
  dft(src_float, dst, cv::DFT_COMPLEX_OUTPUT);
}

void showDFT(const Mat& src){
  Mat splitMat[2] = {Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F)};
  split(src, splitMat);
  Mat dftMagnitude;
  magnitude(splitMat[0], splitMat[1], dftMagnitude);
  dftMagnitude += Scalar::all(1);
  log(dftMagnitude,dftMagnitude);
  normalize(dftMagnitude, dftMagnitude, 0, 1, cv::NORM_MINMAX);

  imshow("DFT", dftMagnitude);
  Mat gray;
  dftMagnitude.convertTo(gray, CV_8U, 255); // upscale to [0..255]
  imwrite("1DFT.jpg", gray);
  waitKey();
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

int main() {

  Mat input_img;
  Mat dft_img;

  input_img = cv::imread("1.jpg", cv::IMREAD_GRAYSCALE);
//input_img = cv::imread("11.jpg", cv::IMREAD_GRAYSCALE);

//  input_img.create(Size(4, 5), cv::IMREAD_GRAYSCALE);
//  for (int i = 0; i < input_img.rows; ++i) {
//    for (int j = 0; j < input_img.cols; ++j) {
//      input_img.at<uint8_t>(i, j) = rand() % 255;
//    }
//  }

  assert(!input_img.empty());
  imshow("input_img", input_img);

  take_USER_DFT(input_img, dft_img);
  //take_OpenCV_DFT(input_img, dft_img);

  Mat imgFromDft;
  //dft(dft_img, imgFromDft, DFT_INVERSE|DFT_REAL_OUTPUT|DFT_SCALE);
  user_invert_dft(dft_img, imgFromDft);
  imshow("imgFromDft", imgFromDft);

  showDFT(dft_img);

  return 0;
}