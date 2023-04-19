#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/aruco.hpp>
#include<opencv2/calib3d.hpp>
#include<iomanip>
#include<sstream>
#include <iostream>
#include <fstream>
#include<ctime>

using namespace std;
using namespace cv;
using namespace cv::aruco;

Ptr<Dictionary> dictionary = getPredefinedDictionary(DICT_6X6_250);
Ptr<aruco::GridBoard> arucoBoard = aruco::GridBoard::create(5, 7, 0.028, 0.007, dictionary);

const float calibrationSquareDimension = 0.0298f;         //размер квадрата на шахм доске [m]
const float arucoSquareDimension = 0.0292f;               //размер маркера                [m]
const Size chessboardDimension = Size(9, 6);

enum calibrateType{
    CHESS_BOARD,
    ARUCO_BOARD
};

void createArucoBoard(const Size boardSize){
  Mat arucoBoardMat;
  arucoBoard->draw(Size(500, 700), arucoBoardMat, 10, 1);
  imwrite("arucoBoard.png", arucoBoardMat);
  imshow("arucoBoard", arucoBoardMat);
  cv::waitKey(0);
}

void createChessBoard(){
  int blockSize = 75;
  Mat chessBoard(chessboardDimension.height * blockSize,chessboardDimension.width * blockSize,CV_8UC3,Scalar::all(0));
  uint8_t color = 0;
  for(int i = 0; i < chessboardDimension.height * blockSize; i += blockSize){
    if (!(chessboardDimension.width & 1)) color=~color;
    for(int j = 0; j < chessboardDimension.width * blockSize; j += blockSize){
      rectangle(chessBoard, Point(j, i),  Point(j + blockSize, i + blockSize), Scalar::all(color), -1);
      color=~color;
    }
  }

  imwrite("chessBoard.png", chessBoard);
  imshow("chessBoard", chessBoard);
  cv::waitKey(0);
}

//создает известные позиции углов для доски
void createKnownBoardPosition(Size boardSize, float squareEdgelength, vector<Point3f>& corners){
  for (int i = 0; i < boardSize.height; i++) {
    for (int j = 0; j < boardSize.width; j++) {
      corners.push_back(Point3f(j * squareEdgelength, i * squareEdgelength, 0.0f));
    }
  }
}

//извлечение из изображения обнаруженных углов шахматной доски (любые углы)
void getChessBoardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false){
  for (auto iter = images.begin(); iter != images.end(); iter++){
    vector<Point2f> pointBuf; //все углы, найденные на 1 изображении
    bool found = findChessboardCorners(*iter, Size(chessboardDimension.width - 1,chessboardDimension.height - 1), pointBuf,
                                       CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    if (found) allFoundCorners.push_back(pointBuf);
    if (showResults){
      drawChessboardCorners(*iter, Size(chessboardDimension.width - 1,chessboardDimension.height - 1), pointBuf, found);
      imshow("Looking for corners", *iter);
      waitKey(0);
    }
  }
}

//извлечение из изображения обнаруженных углов aruco (любые углы)
void getArucoBoardCorners(vector<Mat> images, vector<vector<vector<Point2f>>>& allFoundCorners,
                          vector<vector<int>>& allArucoIds, vector<Size>& imgSize, bool showResults = false){

  Ptr<DetectorParameters> parameters = DetectorParameters::create();
  for (auto iter = images.begin(); iter != images.end(); iter++){
    assert(!(*iter).empty());
    vector<vector<Point2f>> arucoCorners, rejected;
    vector<int> arucoIds;
    detectMarkers(*iter, dictionary, arucoCorners, arucoIds, parameters, rejected);
    allFoundCorners.push_back(arucoCorners);
    allArucoIds.push_back(arucoIds);
    imgSize.push_back((*iter).size());

    if (showResults && !arucoCorners.empty()){
      drawDetectedMarkers(*iter, arucoCorners, arucoIds);
      imshow("Looking for corners on aruco board", *iter);
      waitKey(0);
    }
  }
}

void cameraCalibrationChess(vector<Mat> calibrationImages, Mat& cameraMatrix, Mat& distanceCoefficients)
{
  vector<vector<Point2f>> checkerboardImageSpacePoints;//все найденные углы на всех изображениях
  getChessBoardCorners(calibrationImages, checkerboardImageSpacePoints, false);
  vector<vector<Point3f>> worldSpaseCornerPoints(1);
  createKnownBoardPosition(Size(chessboardDimension.width - 1,chessboardDimension.height - 1), calibrationSquareDimension, worldSpaseCornerPoints[0]);
  worldSpaseCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaseCornerPoints[0]);
  vector<Mat> rVectors, tVectors;
  calibrateCamera(worldSpaseCornerPoints, checkerboardImageSpacePoints, Size(chessboardDimension.width - 1,chessboardDimension.height - 1),
                  cameraMatrix, distanceCoefficients, rVectors, tVectors);
}

void cameraCalibrationAruco(vector<Mat> calibrationImages, Mat& cameraMatrix, Mat& distanceCoefficients)
{
  vector<vector<vector<Point2f>>> allFoundCorners;
  vector<vector<int>> allArucoIds;
  vector<Size> allImgsSize;

  getArucoBoardCorners(calibrationImages, allFoundCorners, allArucoIds, allImgsSize, false);

  vector<vector<Point2f>> allCornersConcatenated;
  vector<int> allIdsConcatenated;
  vector<int> markerCounterPerFrame;
  markerCounterPerFrame.reserve(allFoundCorners.size());

  for (int i = 0; i < allFoundCorners.size(); i++){
    markerCounterPerFrame.push_back(allFoundCorners[i].size());
    for (int j = 0; j < allFoundCorners[i].size(); j++){
      allCornersConcatenated.push_back(allFoundCorners[i][j]);
      allIdsConcatenated.push_back(allArucoIds[i][j]);
    }
  }
  vector<Mat> rvecs, tvecs;
  double repError = cv::aruco::calibrateCameraAruco(allCornersConcatenated, allIdsConcatenated, markerCounterPerFrame, arucoBoard,
                                                    allImgsSize[0], cameraMatrix, distanceCoefficients, rvecs, tvecs, 0);
}

bool saveCameraCalibCoeffs(string name, Mat cameraMatrix, Mat distanceCoefficients)
{
  ofstream outStream(name);
  if (outStream){
    uint16_t rows = cameraMatrix.rows;
    uint16_t columns = cameraMatrix.cols;
    outStream << rows << endl;
    outStream << columns << endl;

    for (int r = 0; r < rows; r++){
      for (int c = 0; c < columns; c++){
        double value = cameraMatrix.at<double>(r, c);
        outStream << value << endl;
      }
    }

    rows = distanceCoefficients.rows;
    columns = distanceCoefficients.cols;
    outStream << rows << endl;
    outStream << columns << endl;

    for (int r = 0; r < rows; r++){
      for (int c = 0; c < columns; c++){
        double value = distanceCoefficients.at<double>(r, c);
        outStream << value << endl;
      }
    }
    outStream.close();
    return true;
  }
  return false;
}

bool loadCameraCalibCoeffs(string name, Mat& cameraMatrix, Mat& distanceCoefficients)
{
  ifstream instream(name);
  if (instream){
    uint16_t rows;
    uint16_t columns;
    instream >> rows;
    instream >> columns;
    cameraMatrix = Mat(Size(columns, rows), CV_64F);

    for (int r = 0; r < rows; r++){
      for (int c = 0; c < columns; c++){
        double read = 0.0f;
        instream >> read;
        cameraMatrix.at<double>(r, c) = read;
        std::cout << cameraMatrix.at<double>(r, c) << "\n";
      }
    }

    //Distance Coefficients
    instream >> rows;
    instream >> columns;
    distanceCoefficients = Mat::zeros(rows, columns, CV_64F);

    for (int r = 0; r < rows; r++){
      for (int c = 0; c < columns; c++){
        double read = 0.0f;
        instream >> read;
        distanceCoefficients.at<double>(r, c) = read;
        std::cout << distanceCoefficients.at<double>(r, c) << "\n";
      }
    }
    instream.close();
    return true;
  }
  return 0;
}

//Функция для калибровки камеры
void cameraCalibrationProcess(Mat& cameraMatrix, Mat& distanceCoefficients, size_t type)
{
  Mat frame;
  Mat drawToFrame;
  vector<Mat> savedImages;
  VideoCapture vid(0);

  assert(vid.isOpened());
  int framePerSecond = 30;
  namedWindow("Camera calibrate", WINDOW_AUTOSIZE);

  while (1){
    bool found = false;
    if (!vid.read(frame)) continue;

    ////CHESS_BOARD
    if(type == CHESS_BOARD){
      vector<Point2f> foundPoints;
      found = findChessboardCorners(frame, Size(chessboardDimension.width - 1,chessboardDimension.height - 1), foundPoints,
                                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
      frame.copyTo(drawToFrame);
      drawChessboardCorners(drawToFrame, Size(chessboardDimension.width - 1,chessboardDimension.height - 1), foundPoints, found);
      if (found) imshow("Camera calibrate", drawToFrame);
      else imshow("Camera calibrate", frame);
    }
    ////ARUCO_BOARD
    else if(type == ARUCO_BOARD){
      Ptr<DetectorParameters> parameters = DetectorParameters::create();
      vector<vector<Point2f>> arucoCorners, rejected;
      vector<int> arucoIds;
      detectMarkers(frame, dictionary, arucoCorners, arucoIds, parameters, rejected);
      frame.copyTo(drawToFrame);
      drawDetectedMarkers(drawToFrame, arucoCorners, arucoIds);
      found = (arucoCorners.size() == arucoBoard->getGridSize().width * arucoBoard->getGridSize().height);
      if (found) imshow("Camera calibrate", drawToFrame);
      else imshow("Camera calibrate", frame);
    } else return;

    char character = waitKey(1000 / framePerSecond);

    switch (character)
    {
      case ' ':
        //saving image
        if (found){
          savedImages.push_back(frame);
          if (type == ARUCO_BOARD){
            imwrite("arucoBoardCalib/" + to_string(savedImages.size()) + ".png", frame);
          }
          else if (type == CHESS_BOARD){
            imwrite("chessBoardCalib/" + to_string(savedImages.size()) + ".png", frame);
          }
          std::cout << savedImages.size() << endl;
        }
        break;

      case 13:
        ////start calibration
        if (savedImages.size() < 10){
          std::cout << "\n[ERROR]. Minimum 10 images need!\n";
          continue;
        }
        else if (type == ARUCO_BOARD) {
          cameraCalibrationAruco(savedImages, cameraMatrix, distanceCoefficients);
          saveCameraCalibCoeffs("aruco_сameraCalibCoeffs", cameraMatrix, distanceCoefficients);
        }
        else if (type == CHESS_BOARD) {
          cameraCalibrationChess(savedImages, cameraMatrix, distanceCoefficients);
          saveCameraCalibCoeffs("chess_сameraCalibCoeffs", cameraMatrix, distanceCoefficients);
        } else return;

        std::cout << "\nCalibration success!\n";
        return;
      case 27:
        //exit
        return;
    }
  }
}

void drawCube(Mat inputImage, Mat cameraMatrix, Mat distCoeffs, Vec3d rvecs, Vec3d tvecs, float len)
{
  vector<Point2f> imagePoints;
  vector<Point3f> pointWorld(8, Point3d(0, 0, 0));
  pointWorld[0] = Point3d(len / 2, len / 2, 0);
  pointWorld[1] = Point3d(-len / 2, len / 2, 0);
  pointWorld[2] = Point3d(-len / 2, -len / 2, 0);
  pointWorld[3] = Point3d(len / 2, -len / 2, 0);

  pointWorld[4] = Point3d(len / 2, len / 2, len);
  pointWorld[5] = Point3d(-len / 2, len / 2, len);
  pointWorld[6] = Point3d(-len / 2, -len / 2, len);
  pointWorld[7] = Point3d(len / 2, -len / 2, len);

  projectPoints(pointWorld, rvecs, tvecs, cameraMatrix, distCoeffs, imagePoints);

  // красные линии
  for (int i = 0; i < 8; i+=2){
    line(inputImage, imagePoints[i], imagePoints[i + 1], Scalar(0, 0, 255), 2);
  }

  // зеленые линии
  line(inputImage, imagePoints[1], imagePoints[2], Scalar(0, 255, 0), 2);
  line(inputImage, imagePoints[3], imagePoints[0], Scalar(0, 255, 0), 2);
  line(inputImage, imagePoints[5], imagePoints[6], Scalar(0, 255, 0), 2);
  line(inputImage, imagePoints[7], imagePoints[4], Scalar(0, 255, 0), 2);

  // синии линии
  for (int i = 0; i < 4; i++){
    line(inputImage, imagePoints[i], imagePoints[i + 4], Scalar(255, 0, 0), 2);
  }
}

int startWebcamMonitoring(const Mat& cameraMatrix, Mat& distanceCoefficients)
{
  Mat frame;
  vector<int> arucoIds;
  vector<vector<Point2f>> arucoCorners, rejected;
  Ptr<DetectorParameters> parameters = DetectorParameters::create();
  VideoCapture vid(0);
  assert(vid.isOpened());

  namedWindow("Camera", WINDOW_AUTOSIZE);
  vector<Vec3d> rotationVectors, translationVectors; //векторы вращения и перемещения(x,y,z) распознанных маркеров относительно камеры
  while (true)
  {
    if (!vid.read(frame)) continue;
    detectMarkers(frame, dictionary, arucoCorners, arucoIds, parameters, rejected);
    //функция определения позиции и ориентации распознанных маркеров относительно камеры
    aruco::estimatePoseSingleMarkers(arucoCorners, arucoSquareDimension,
                                     cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);
    for (int i = 0; i < arucoIds.size(); i++){
      //drawDetectedMarkers(frame, arucoCorners, arucoIds);
      //aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.025f); //рисование осей маркеров
      drawCube(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], arucoSquareDimension);
    }
    imshow("Camera", frame);
    if (waitKey(30) >= 0) break;
  }
  return 1;
}

int main()
{
  Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
  Mat distanceCoefficients = Mat::zeros(8, 1, CV_64F);

  ////create aruco board
  //const Size boardSize = Size(500, 700);
  //createArucoBoard(boardSize);

  ////create chess board
  //createChessBoard();

  ////calibrate by chess board
  //cameraCalibrationProcess(cameraMatrix, distanceCoefficients, CHESS_BOARD);

  ////calibrate by aruco board
  //cameraCalibrationProcess(cameraMatrix, distanceCoefficients, ARUCO_BOARD);

  ////load camera params
  //loadCameraCalibCoeffs("chess_сameraCalibCoeffs", cameraMatrix, distanceCoefficients);
  //loadCameraCalibCoeffs("aruco_сameraCalibCoeffs", cameraMatrix, distanceCoefficients);
  loadCameraCalibCoeffs("matlab_сameraCalibCoeffs", cameraMatrix, distanceCoefficients);

  ////find markers and draw cube
  startWebcamMonitoring(cameraMatrix, distanceCoefficients);

  return 0;
}