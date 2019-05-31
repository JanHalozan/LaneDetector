//
//  main.cpp
//  LaneDetector
//
//  Created by Jan Halozan on 16/05/2019.
//  Copyright © 2019 JanHalozan. All rights reserved.
//

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

tuple<Mat, Mat, vector<Mat>, vector<Mat>> calibrateCameraWithCheckerboard(String folder, int picturesCount)
{
    const float squareSize = 20.0f; //in mm - https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf
    const int verticalCorners = 9;
    const int horizontalCorners = 7;
    Size boardSize(horizontalCorners, verticalCorners);
    
    vector<String> files;
    for (int i = 0; i < picturesCount; ++i)
        files.push_back(folder + to_string(i + 1) + ".jpg");
    
    vector<vector<Point3f>> objectPoints;
    vector<vector<Point2f>> imagePoints;
    vector<Point2f> corners;
    
    Mat img, gray;
    
    for (int i = 0; i < files.size(); ++i)
    {
        img = imread(files[i]);
        cvtColor(img, gray, COLOR_BGR2GRAY);
        
        bool found = findChessboardCorners(img, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
        
        if (!found)
        {
            cout << "Could not find corners for image." << files[i];
        }
        
        cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
        drawChessboardCorners(gray, boardSize, corners, found);
        
        vector<Point3f> obj;
        for (int y = 0; y < verticalCorners; ++y)
            for (int x = 0; x < horizontalCorners; ++x)
                obj.push_back(Point3f((float)x * squareSize, (float)y * squareSize, 0.0f));
        
        imagePoints.push_back(corners);
        objectPoints.push_back(obj);
    }
    
    Mat cameraMatrix;
    Mat distortionCoefficients;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    
    calibrateCamera(objectPoints, imagePoints, img.size(), cameraMatrix, distortionCoefficients, rvecs, tvecs);
    
    return make_tuple(cameraMatrix, distortionCoefficients, rvecs, tvecs);
}

vector<Point2f> slidingWindow(Mat image, Rect window)
{
    vector<Point2f> points;
    const Size imgSize = image.size();
    bool shouldBreak = false;
    
    while (true)
    {
        float currentX = window.x + window.width * 0.5f;
        
        Mat roi = image(window); //Extract region of interest
        vector<Point2f> locations;
        
        findNonZero(roi, locations); //Get all non-black pixels. All are white in our case
        
        float avgX = 0.0f;
        
        for (int i = 0; i < locations.size(); ++i) //Calculate average X position
        {
            float x = locations[i].x;
            avgX += window.x + x;
        }
        
        avgX = locations.empty() ? currentX : avgX / locations.size();
        
        Point point(avgX, window.y + window.height * 0.5f);
        points.push_back(point);
        
        //Move the window up
        window.y -= window.height;
        
        //For the uppermost position
        if (window.y < 0)
        {
            window.y = 0;
            shouldBreak = true;
        }
        
        //Move x position
        window.x += (point.x - currentX);
        
        //Make sure the window doesn't overflow, we get an error if we try to get data outside the matrix
        if (window.x < 0)
            window.x = 0;
        if (window.x + window.width >= imgSize.width)
            window.x = imgSize.width - window.width - 1;
        
        if (shouldBreak)
            break;
    }
    
    return points;
}

int main(int argc, char** argv)
{
    //If you need it it's here
//    const String calibFolder = "/path/to/calibration/folder/";
//    const int calibImages = 6; //Number of calib images
//    auto calibrationData = calibrateCameraWithCheckerboard(calibFolder, calibImages);
    
    const String filename = "path/to/your/clip/ride1.mov";
    VideoCapture cap(filename);
    
    if (!cap.isOpened())
    {
        cout << "Could not open video stream." << endl;
        return 1;
    }
    
    //Prepare things that don't need to be computed on every frame.
    Point2f srcVertices[4];
    
    //Define points that are used for generating bird's eye view. This was done by trial and error. Best to prepare sliders and configure for each use case.
    //    srcVertices[0] = Point(790, 605); //These were another test.
    //    srcVertices[1] = Point(900, 605);
    //    srcVertices[2] = Point(1760, 1030);
    //    srcVertices[3] = Point(150, 1030);
    
    srcVertices[0] = Point(700, 605);
    srcVertices[1] = Point(890, 605);
    srcVertices[2] = Point(1760, 1030);
    srcVertices[3] = Point(20, 1030);
    
    //Destination vertices. Output is 640 by 480px
    Point2f dstVertices[4];
    dstVertices[0] = Point(0, 0);
    dstVertices[1] = Point(640, 0);
    dstVertices[2] = Point(640, 480);
    dstVertices[3] = Point(0, 480);
    
    //Prepare matrix for transform and get the warped image
    Mat perspectiveMatrix = getPerspectiveTransform(srcVertices, dstVertices);
    Mat dst(480, 640, CV_8UC3); //Destination for warped image
    
    //For transforming back into original image space
    Mat invertedPerspectiveMatrix;
    invert(perspectiveMatrix, invertedPerspectiveMatrix);
    
    
    Mat org; //Original image, modified only with result
    Mat img; //Working image
    
    while (true)
    {
        //Read a frame
        cap.read(org);
        if (org.empty()) //When this happens we've reached the end
            break;
        
        //Generate bird's eye view
        warpPerspective(org, dst, perspectiveMatrix, dst.size(), INTER_LINEAR, BORDER_CONSTANT);
        
        //Convert to gray
        cvtColor(dst, img, COLOR_RGB2GRAY);
        
        //Extract yellow and white info
        Mat maskYellow, maskWhite;
        
        inRange(img, Scalar(20, 100, 100), Scalar(30, 255, 255), maskYellow);
        inRange(img, Scalar(150, 150, 150), Scalar(255, 255, 255), maskWhite);
        
        Mat mask, processed;
        bitwise_or(maskYellow, maskWhite, mask); //Combine the two masks
        bitwise_and(img, mask, processed); //Extrect what matches
        
        
        //Blur the image a bit so that gaps are smoother
        const Size kernelSize = Size(9, 9);
        GaussianBlur(processed, processed, kernelSize, 0);
        
        //Try to fill the gaps
        Mat kernel = Mat::ones(15, 15, CV_8U);
        dilate(processed, processed, kernel);
        erode(processed, processed, kernel);
        morphologyEx(processed, processed, MORPH_CLOSE, kernel);
        
        //Keep only what's above 150 value, other is then black
        const int thresholdVal = 150;
        threshold(processed, processed, thresholdVal, 255, THRESH_BINARY);
        //Might be optimized with adaptive thresh
        
        
        //Get points for left sliding window. Optimize by using a histogram for the starting X value
        vector<Point2f> pts = slidingWindow(processed, Rect(0, 420, 120, 60));
        vector<Point> allPts; //Used for the end polygon at the end.
        
        /* potential hist calculation for optimization. Should be moved above slidingWindow call and used for determining X value
         Mat hist;
         int histSize = 256;
         float range[] = { 0, 256 }; //the upper boundary is exclusive
         const float* histRange = { range };
         calcHist(&processed, 1, 0, Mat(), hist, 1, &histSize, &histRange, true);
         */
        
        vector<Point2f> outPts;
        perspectiveTransform(pts, outPts, invertedPerspectiveMatrix); //Transform points back into original image space
        
        //Draw the points onto the out image
        for (int i = 0; i < outPts.size() - 1; ++i)
        {
            line(org, outPts[i], outPts[i + 1], Scalar(255, 0, 0), 3);
            allPts.push_back(Point(outPts[i].x, outPts[i].y));
        }
        
        allPts.push_back(Point(outPts[outPts.size() - 1].x, outPts[outPts.size() - 1].y));
        
        Mat out;
        cvtColor(processed, out, COLOR_GRAY2BGR); //Conver the processing image to color so that we can visualise the lines
        
        for (int i = 0; i < pts.size() - 1; ++i) //Draw a line on the processed image
            line(out, pts[i], pts[i + 1], Scalar(255, 0, 0));
        
        //Sliding window for the right side
        pts = slidingWindow(processed, Rect(520, 420, 120, 60));
        perspectiveTransform(pts, outPts, invertedPerspectiveMatrix);
        
        //Draw the other lane and append points
        for (int i = 0; i < outPts.size() - 1; ++i)
        {
            line(org, outPts[i], outPts[i + 1], Scalar(0, 0, 255), 3);
            allPts.push_back(Point(outPts[outPts.size() - i - 1].x, outPts[outPts.size() - i - 1].y));
        }
        
        allPts.push_back(Point(outPts[0].x - (outPts.size() - 1) , outPts[0].y));
        
        for (int i = 0; i < pts.size() - 1; ++i)
            line(out, pts[i], pts[i + 1], Scalar(0, 0, 255));

        //Create a green-ish overlay
        vector<vector<Point>> arr;
        arr.push_back(allPts);
        Mat overlay = Mat::zeros(org.size(), org.type());
        fillPoly(overlay, arr, Scalar(0, 255, 100));
        addWeighted(org, 1, overlay, 0.5, 0, org); //Overlay it
        
        //Show results
        imshow("Preprocess", out);
        imshow("src", org);
        
        if (waitKey(50) > 0)
            break;
    }
    
    
    cap.release();
    
    waitKey();
    return 0;
}
