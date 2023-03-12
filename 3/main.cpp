#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>
#include <cvDirectory.h>
#include <cmath>

const int max_value_H = 179;
const int max_value = 255;
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value;

void trackBarHsv(cv::Mat &img_hsv) {
  cv::Mat img_threshold;
  //create trackbars
  cv::namedWindow("Trackbars", cv::WINDOW_AUTOSIZE);
  cv::createTrackbar("Hue min", "Trackbars", &low_H, max_value_H);
  cv::createTrackbar("Hue max", "Trackbars", &high_H, max_value_H);
  cv::createTrackbar("Sat min", "Trackbars", &low_S, max_value);
  cv::createTrackbar("Sat max", "Trackbars", &high_S, max_value);
  cv::createTrackbar("Val min", "Trackbars", &low_V, max_value);
  cv::createTrackbar("Val max", "Trackbars", &high_V, max_value);

  while (1) {
    // Detect the object based on HSV Range Values
    inRange(img_hsv, cv::Scalar(low_H, low_S, low_V), cv::Scalar(high_H, high_S, high_V), img_threshold);
    //show images
    cv::imshow("Trackbars", img_threshold);
    cv::waitKey(1);
  }
}

void do_part_1() {
  std::string addrImages = "../LAB3/img_zadan/allababah/";
  std::vector<std::string> imagesData = cv::Directory::GetListFiles(addrImages);
  assert(!imagesData.empty());
  for (size_t imageIndex = 0; imageIndex < imagesData.size(); imageIndex++) {
    //read image
    cv::Mat image = cv::imread(addrImages + imagesData[imageIndex]);
    assert(!image.empty());

    //create img_gray, img_bin
    cv::Mat img_gray;
    cv::Mat img_bin;
    cv::Mat img_res;
    cv::Mat erodeMat;
    cv::Mat imgContour;

    image.copyTo(img_res);
    cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);

    //binary
    cv::threshold(img_gray, img_bin, 200, 255, cv::THRESH_BINARY);

    //delete noise
    cv::Mat kernel(cv::Size(7, 7), CV_8UC1, cv::Scalar(255));
    cv::erode(img_bin, img_bin, kernel);
    cv::dilate(img_bin, img_bin, kernel);

    cv::erode(img_bin, erodeMat, kernel);

    imgContour = img_bin - erodeMat;

    //show images
    cv::imshow("imgContour: " + imagesData[imageIndex], imgContour);
    cv::imshow("img_bin: " + imagesData[imageIndex], img_bin);
    cv::waitKey();

    //find contours
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(img_bin.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    //draw contour's central points
    for (size_t i = 0; i < contours.size(); i++) {
      cv::Moments moments = cv::moments(contours[i]);
      double m00 = moments.m00;
      double m01 = moments.m01;
      double m10 = moments.m10;
      cv::Point center_pointer(m10 / m00, m01 / m00);
      cv::circle(img_res, center_pointer, 2, cv::Scalar(0, 0, 255), -1);
      polylines(img_res, contours[i], true, cv::Scalar(0, 255, 0), 2, 8);
    }

    //show images
    cv::imshow("RES: " + imagesData[imageIndex], img_res);
    cv::waitKey();
  }
}

void do_part_2() {
  std::string addrImages = "../LAB3/img_zadan/teplovizor/";
  std::vector<std::string> imagesData = cv::Directory::GetListFiles(addrImages);
  assert(!imagesData.empty());
  for (size_t imageIndex = 0; imageIndex < imagesData.size(); imageIndex++) {
    //read image
    cv::Mat image = cv::imread(addrImages + imagesData[imageIndex]);
    assert(!image.empty());

    //create img_hsv
    cv::Mat img_hsv;
    cv::Mat img_res;
    cv::Mat img_threshold;
    cv::Mat tmp_img_threshold;
    cv::Mat erodeMat;
    cv::Mat imgContour;
    image.copyTo(img_res);

    // Convert from BGR to HSV colorspace
    cvtColor(image, img_hsv, cv::COLOR_BGR2HSV);

    // trackBarHsv(img_hsv);

    // Detect the object based on HSV Range Values
    inRange(img_hsv, cv::Scalar(149, 0, 0), cv::Scalar(179, 255, 255), tmp_img_threshold);
    inRange(img_hsv, cv::Scalar(0, 0, 0), cv::Scalar(50, 255, 255), img_threshold);
    img_threshold += tmp_img_threshold;

    //delete noise
    cv::Mat kernel(cv::Size(7, 7), CV_8UC1, cv::Scalar(255));
    cv::erode(img_threshold, img_threshold, kernel);
    cv::dilate(img_threshold, img_threshold, kernel);

    cv::erode(img_threshold, erodeMat, kernel);
    imgContour = img_threshold - erodeMat;

    //show images
    cv::imshow("imgContour: " + imagesData[imageIndex], imgContour);
    cv::imshow("img_threshold: " + imagesData[imageIndex], img_threshold);
    cv::waitKey();


    //find contours
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(img_threshold.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    //draw contour's central points
    for (size_t i = 0; i < contours.size(); i++) {
      cv::Moments moments = cv::moments(contours[i]);
      double m00 = moments.m00;
      double m01 = moments.m01;
      double m10 = moments.m10;
      cv::Point center_pointer(m10 / m00, m01 / m00);
      cv::circle(img_res, center_pointer, 3, cv::Scalar(0, 0, 0), -1);
    }

    //show images
    cv::imshow("RES: " + imagesData[imageIndex], img_res);
    cv::waitKey();
  }
}

bool contourIsValid(std::vector<cv::Point> contour, size_t k, cv::Mat &img_res, cv::Point lampCenterPointer) {

  cv::Moments contMoment = cv::moments(contour);
  double m00 = contMoment.m00;
  double m01 = contMoment.m01;
  double m10 = contMoment.m10;
  cv::Point contCenter(m10 / m00, m01 / m00);

  double dst2Lamp = (double) sqrt(pow(lampCenterPointer.x - contCenter.x, 2) +
                                  pow(lampCenterPointer.y - contCenter.y, 2));

  cv::Point right = contour[0];
  cv::Point left = contour[0];
  cv::Point up = contour[0];
  cv::Point down = contour[0];

  for (int i = 0; i < contour.size(); i++) {
    if (contour[i].x > right.x) right = contour[i];
    if (contour[i].x < left.x) left = contour[i];
    if (contour[i].y > up.y) up = contour[i];
    if (contour[i].y < down.y) down = contour[i];
  }

  double widthContour = abs(right.x - left.x);
  double heightContour = abs(up.y - down.y);

  if (contourArea(contour) < 200) return false;
  if (dst2Lamp < 50) return false;
  if ((double) widthContour / heightContour >= 3) return false;
  if ((double) heightContour / widthContour >= 3) return false;
  if ((double) (up.x - left.x) / (right.x - up.x) >= 30) return false;
  if ((double) (right.x - up.x) / (up.x - left.x) >= 30) return false;
  if ((double) (down.x - left.x) / (right.x - down.x) >= 30) return false;
  if ((double) (right.x - down.x) / (down.x - left.x) >= 30) return false;
  if ((double) (up.y - left.y) / (left.y - down.y) >= 30) return false;
  if ((double) (left.y - down.y) / (up.y - left.y) >= 30) return false;
  if ((double) (up.y - right.y) / (right.y - down.y) >= 30) return false;
  if ((double) (right.y - down.y) / (up.y - right.y) >= 30) return false;
  return true;
}

void do_part_3() {
  std::string addrImages = "../LAB3/img_zadan/roboti/";
  std::vector<std::string> imagesData = cv::Directory::GetListFiles(addrImages);
  assert(!imagesData.empty());
  for (size_t imageIndex = 0; imageIndex < imagesData.size(); imageIndex++) {
    //read image
    cv::Mat image = cv::imread(addrImages + imagesData[imageIndex]);
    assert(!image.empty());

    //create img
    cv::Mat img_hsv;
    cv::Mat img_res;
    cv::Mat img_bin;
    cv::Mat img_threshold;
    cv::Mat img_gray;
    image.copyTo(img_res);

    //<-----find lamp----->//
    //to gray
    cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);

    //binary
    cv::threshold(img_gray, img_bin, 250, 255, cv::THRESH_BINARY);

    //delete noise
    cv::Mat kernel(cv::Size(7, 7), CV_8UC1, cv::Scalar(255));
    cv::erode(img_bin, img_bin, kernel);
    cv::dilate(img_bin, img_bin, kernel);

    //find contours
    std::vector<std::vector<cv::Point> > contoursLamp;
    cv::findContours(img_bin.clone(), contoursLamp, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::drawContours(img_res, contoursLamp, -1, cv::Scalar(0, 255, 255), 3);

//    //show images
//    cv::imshow("RES: " + imagesData[imageIndex], img_res);
//    cv::waitKey();


    //get lamp's coordinates
    assert(contoursLamp.size() == 1);
    cv::Moments lampCtrMoment = cv::moments(contoursLamp[0]);
    double m00 = lampCtrMoment.m00;
    double m01 = lampCtrMoment.m01;
    double m10 = lampCtrMoment.m10;
    cv::Point lampCenterPointer(m10 / m00, m01 / m00);

    //<-----find caps----->//
    // Convert from BGR to HSV colorspace
    cvtColor(image, img_hsv, cv::COLOR_BGR2HSV);

    //trackBarHsv(img_hsv);

    //get threshold parameters
    std::vector<cv::Scalar> lowBrd = {cv::Scalar(0, 60, 125),    // R
                                      cv::Scalar(50, 60, 125),    // G
                                      cv::Scalar(80, 60, 125)};   // B

    std::vector<cv::Scalar> upBrd = {cv::Scalar(50, 255, 255),     // R
                                     cv::Scalar(80, 255, 255),      // G
                                     cv::Scalar(115, 255, 255)};    // B

    std::vector<cv::Scalar> contoursColorBGR = {cv::Scalar(0, 0, 255),    // R
                                                cv::Scalar(0, 255, 0),    // G
                                                cv::Scalar(255, 0, 0)};   // B

    for (size_t i = 0; i < contoursColorBGR.size(); i++) {
      // Detect the object based on HSV Range Values
      //for red color
      if (i == 0) {
        cv::Mat tmp_img_threshold;
        inRange(img_hsv, cv::Scalar(129, 60, 125), cv::Scalar(179, 255, 255), tmp_img_threshold);
        inRange(img_hsv, lowBrd[i], upBrd[i], img_threshold);
        img_threshold += tmp_img_threshold;
      } else {
        inRange(img_hsv, lowBrd[i], upBrd[i], img_threshold);
      }

      //delete noise
      cv::Mat kernel(cv::Size(7, 7), CV_8UC1, cv::Scalar(255));
      cv::erode(img_threshold, img_threshold, kernel);
      cv::dilate(img_threshold, img_threshold, kernel);

      cv::Mat erodeMat;
      cv::Mat imgContour;

      cv::erode(img_threshold, erodeMat, kernel);
      imgContour = img_threshold - erodeMat;

//      //show images
//      cv::imshow("img_threshold: " + imagesData[imageIndex], img_threshold);
//      cv::imshow("imgContour: " + imagesData[imageIndex], imgContour);
//      cv::waitKey();

      //find contours
      std::vector<std::vector<cv::Point> > contours;
      cv::findContours(img_threshold.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
      // cv::drawContours(img_res, contours, -1, contoursColorBGR[i], 3);

      //find nearest robot to lamp
      size_t minDist2Lamp = 0;
      cv::Point minDist2LampPoint;
      bool is_first = true;
      for (size_t k = 0; k < contours.size(); k++) {

        if (!contourIsValid(contours[k], k, img_res, lampCenterPointer)) continue;

        polylines(img_res, contours[k], true, contoursColorBGR[i], 3, 8);

        cv::Moments moments = cv::moments(contours[k]);
        double m00 = moments.m00;
        double m01 = moments.m01;
        double m10 = moments.m10;
        cv::Point robotCenterPointer(m10 / m00, m01 / m00);
        double dst2Lamp = (double) sqrt(pow(lampCenterPointer.x - robotCenterPointer.x, 2) +
                                        pow(lampCenterPointer.y - robotCenterPointer.y, 2));
        if (is_first) {
          minDist2Lamp = dst2Lamp;
          minDist2LampPoint = robotCenterPointer;
          is_first = false;
        } else if (minDist2Lamp > dst2Lamp) {
          minDist2Lamp = dst2Lamp;
          minDist2LampPoint = robotCenterPointer;
        }
      }
      cv::circle(img_res, minDist2LampPoint, 5, cv::Scalar(0, 0, 0), -1);
    }

    //show images
    cv::imshow("RES: " + imagesData[imageIndex], img_res);
    cv::waitKey();
  }
}

void do_part_4() {
  std::string addrImages = "../LAB3/img_zadan/gk/";
  std::vector<std::string> imagesData = cv::Directory::GetListFiles(addrImages);
  assert(!imagesData.empty());

  std::vector<std::vector<cv::Point> > wrenchContour;

  //find tmplts moments's character
  for (size_t imageIndex = 0; imageIndex < imagesData.size(); imageIndex++) {
    if (imagesData[imageIndex] != "gk_tmplt.jpg") continue;

    //read image
    cv::Mat image = cv::imread(addrImages + imagesData[imageIndex]);
    assert(!image.empty());

    //create img_gray, img_bin
    cv::Mat img_gray;
    cv::Mat img_bin;
    cv::Mat img_res;
    image.copyTo(img_res);
    cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);

    //binary
    cv::threshold(img_gray, img_bin, 200, 255, cv::THRESH_BINARY);

    //delete noise
    cv::Mat kernel(cv::Size(7, 7), CV_8UC1, cv::Scalar(255));
    cv::erode(img_bin, img_bin, kernel);
    cv::dilate(img_bin, img_bin, kernel);

    //find contours
    cv::findContours(img_bin.clone(), wrenchContour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::drawContours(img_res, wrenchContour, -1, cv::Scalar(0, 255, 255), 2);
    assert(wrenchContour.size() == 1);
  }

  for (size_t imageIndex = 0; imageIndex < imagesData.size(); imageIndex++) {
    if (imagesData[imageIndex] == "gk_tmplt.jpg") continue;

    //read image
    cv::Mat image = cv::imread(addrImages + imagesData[imageIndex]);
    assert(!image.empty());

    //create img_gray, img_bin
    cv::Mat img_gray;
    cv::Mat img_bin;
    cv::Mat img_res;
    image.copyTo(img_res);
    cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);

    //binary
    cv::threshold(img_gray, img_bin, 240, 255, cv::THRESH_BINARY_INV);

    //delete noise
    cv::Mat kernel(cv::Size(7, 7), CV_8UC1, cv::Scalar(255));
    cv::erode(img_bin, img_bin, kernel);
    cv::dilate(img_bin, img_bin, kernel);

    cv::Mat erodeMat;
    cv::Mat imgContour;

    cv::erode(img_bin, erodeMat, kernel);

    imgContour = img_bin - erodeMat;

    //show images
    cv::imshow("imgContour: " + imagesData[imageIndex], imgContour);
    cv::imshow("img_bin: " + imagesData[imageIndex], img_bin);
    cv::waitKey();

    //find contours
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(img_bin.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    //draw contour's central points
    for (size_t i = 0; i < contours.size(); i++) {
      double diffContours = cv::matchShapes(wrenchContour[0], contours[i], cv::CONTOURS_MATCH_I2, 0);
      if (diffContours > 0.5) {
        polylines(img_res, contours[i], true, cv::Scalar(0, 0, 255), 3, 8);
      } else {
        polylines(img_res, contours[i], true, cv::Scalar(0, 255, 0), 3, 8);
      }
    }

    //show images
    cv::imshow("RES: " + imagesData[imageIndex], img_res);
    cv::waitKey();
  }
}

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cerr << "[program_name][task's part (0...4)]" << std::endl;
    return -1;
  }

  std::string assigmentPart = argv[1];

  switch (std::stoi(assigmentPart)) {
    case 1: {
      do_part_1();
      break;
    }
    case 2: {
      do_part_2();
      break;
    }
    case 3: {
      do_part_3();
      break;
    }
    case 4: {
      do_part_4();
      break;
    }
    default: {
      std::cerr << "[ERROR]: invalid part. Choose 1...4.\n";
      exit(-1);
    }
  }
  return 0;
}