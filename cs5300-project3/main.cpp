/**
 * Kaelyn Jefferson
 * CS5300 Project 3
 * Main that calls object recognition functions used for project 3.
 */

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "ORfxns.cpp"
#include "ORfxns.h"
using namespace cv;

/**
 * @brief Main function starts video/photo input and listens for user key functions in the Object recognition system.
 *
 * @param argc 2 values
 * @param argv "cs5300-project3 photo", "cs5300-project3 video"
 * @return int
 */
int main(int argc, const char *argv[])
{

    std::string currentFeatures = "";
    std::string pNearestNeighbors = "";

    // does OR techniques on live video
    if (std::string(argv[1]) == ("video"))
    {
        cv::VideoCapture *capdev;
        cv::Mat dst;

        // capture video frame
        capdev = new cv::VideoCapture(0);
        if (!capdev->isOpened())
        {
            printf("Unable to open video for you\n");
            return (-1);
        }

        // get the properties of the image
        cv::Size refS((int)capdev->get(cv::CAP_PROP_FRAME_WIDTH),
                      (int)capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

        cv::namedWindow("Video", 1);
        cv::Mat frame;
        int k = 0;
        int setOnce = 0;
        std::string objectLabel;

        for (;;)
        {
            if (k == 0)
            {
                delete capdev;
                capdev = new cv::VideoCapture(0);
                resetDistanceMetrics();
                setOnce = 0;
                currentFeatures = "";
                objectLabel = "";
                pNearestNeighbors = "";
            }
            *capdev >> frame;

            if (k >= 1)
            {
                thresholdAlgo(frame, frame);
                for (int p = 0; p < 11; p++)
                {
                    dilatePhoto(frame, frame);
                }
                for (int i = 0; i < 7; i++)
                {
                    erodePhoto(frame, frame);
                }
            }
            if (k >= 2)
            {
                cv::Mat regionMap = applyConnectedComponentsAnalysis(frame, frame);
                std::string featureString = regionFeatures(regionMap, frame, 1);
                if (setOnce == 0)
                {
                    currentFeatures.append(featureString);
                    setOnce += 1;
                }
                std::string featuresToDisplay = "Features:" + featureString;
                cv::putText(frame, featuresToDisplay, Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(54, 117, 136), 2);
            }
            if (k == 3)
            {
                cv::String nameSpecified;
                std::cout << "Name/Label for current object: ";
                std::cin >> nameSpecified;
                storeFeatures(nameSpecified, currentFeatures);
                k = 4;
            }
            if (k == 5)
            {
                objectLabel = classNewImages(currentFeatures);
                std::cout << "\nScaled Euclidian classified as: " + objectLabel + "\n"
                          << std::endl;
                k = 6;
            }
            if (k >= 5)
            {
                std::string labelToDisplay = "New image classified as: " + objectLabel;
                cv::putText(frame, labelToDisplay, Point(20, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(54, 117, 136), 2);
            }
            if (k == 7)
            {
                // gets the current matrix from the file
                int confusionMatrix[10][10] = {};

                std::ifstream myfile;
                myfile.open("ConfusionMatrix.txt");
                std::string myline;
                if (myfile.is_open())
                {
                    int row = 0;
                    while (myfile)
                    {
                        std::getline(myfile, myline);
                        std::string s = myline;
                        for (int col = 0; col < 10; col++)
                        {
                            int num = atof(s.substr(0, 1).c_str());
                            confusionMatrix[row][col] = num;
                            s.erase(0, 1);
                        }
                        row += 1;
                    }
                }
                myfile.close();

                int classified = 0;
                int truth = 0;

                pNearestNeighbors = kNearestNeighbor(2, currentFeatures);
                std::cout << "\nKNN says this is the object: " + pNearestNeighbors << std::endl;

                classified = getClassified(pNearestNeighbors);

                // ask user what this is, then add tally
                cv::String nameSpecified;
                std::cout << "The true name for current object: ";
                std::cin >> nameSpecified;

                truth = getTruth(nameSpecified);

                // adds to the current matrix and then stores that into a file
                confusionMatrix[classified][truth]++;
                std::cout << "The Confusion Matrix:" << std::endl;

                for (int i = 0; i < 10; i++)
                {
                    for (int j = 0; j < 10; j++)
                    {
                        std::cout << confusionMatrix[i][j];
                    }
                    std::cout << std::endl;
                }

                // save in a file
                storeConfusionMatrix(confusionMatrix);
                k = 8;
            }
            if (k >= 8)
            {
                std::string KNNLabel = "KNN classified as: " + pNearestNeighbors;
                cv::putText(frame, KNNLabel, Point(20, 80), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(54, 117, 136), 2);
            }

            cv::imshow("Video", frame);
            char key = cv::waitKey(10);

            if (key == 'q')
            {
                break;
            }
            else if (key == 't') // threshold algorithm + clean up binary (Q1, Q2)
            {
                k = 1;
            }
            else if (key == 's') // saving an image
            {
                imwrite("/Users/kaelynjefferson/Desktop/sampleImage.jpg", frame);
                cv::imshow("Video", frame);
            }

            else if (key == 'f') // segment into regions (Q3, Q4)
            {
                k = 2;
            }
            else if (key == 'n') // collect training data (Q5)
            {
                k = 3;
            }
            else if (key == 'l') // classify new images (Q6)
            {
                k = 5;
            }
            else if (key == 'k') // implement the KNN classifier (Q7), adds and prints confusion matrix (Q8)
            {
                k = 7;
            }
            else if (key == 'r') // resets the video to start another object
            {
                k = 0;
            }
        }
        delete capdev;
        return 0;
    }

    if (std::string(argv[1]) == ("photo"))
    {
        cv::Mat img = imread("/Users/kaelynjefferson/Documents/NEU/MSCS/MSCS semesters/2022 Spring/cs5300-project3/TenObjects/unknown10.jpg", cv::IMREAD_COLOR);
        cv::Mat dst;
        std::string currentFeatures;

        if (img.empty())
        {
            std::cout << "could not read the image!" << std::endl;
            return 1;
        }
        imshow("Display window", img);
        int k = cv::waitKey(0);

        while (k != 'q')
        {
            if (k == 't') // threshold algorithm (Q1)
            {
                resetDistanceMetrics();
                thresholdAlgo(img, dst);
            }
            else if (k == 'd') // cleaning up binary (Q2)
            {
                dilatePhoto(dst, dst);
            }
            else if (k == 'e') // cleaning up binary (Q2)
            {
                erodePhoto(dst, dst);
            }
            else if (k == 's') // save the image for testing purposes
            {
                imwrite("/Users/kaelynjefferson/Desktop/sampleImageThresholded.jpg", dst);
            }
            else if (k == 'f') // segmenting the image into regions, compute features for each region (Q3, Q4)
            {
                cv::Mat regionMap = applyConnectedComponentsAnalysis(dst, dst);
                std::string featureString = regionFeatures(regionMap, dst, 1);
                currentFeatures.append(featureString);
            }
            else if (k == 'n') // collect training data (Q5)
            {
                cv::String nameSpecified;
                std::cout << "Name/Label for current object: ";
                std::cin >> nameSpecified;
                storeFeatures(nameSpecified, currentFeatures);
            }
            else if (k == 'l') // classify new images (Q6)
            {
                std::string objectLabel = classNewImages(currentFeatures);
                std::cout << "Scaled Euclidian classified as: " + objectLabel + "\n"
                          << std::endl;
                // std::cout << "\nNew image is classified as: " + objectLabel << std::endl;
                cv::putText(dst, objectLabel, Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(54, 117, 136), 2);
            }
            else if (k == 'k') // implement the KNN classifier (Q7), adds and prints confusion matrix (Q8)
            {

                // gets the current matrix from the file
                int confusionMatrix[10][10] = {};

                std::ifstream myfile;
                myfile.open("ConfusionMatrix.txt");
                std::string myline;
                if (myfile.is_open())
                {
                    int row = 0;
                    while (myfile)
                    {
                        std::getline(myfile, myline);
                        std::string s = myline;
                        for (int col = 0; col < 10; col++)
                        {
                            int num = atof(s.substr(0, 1).c_str());
                            confusionMatrix[row][col] = num;
                            s.erase(0, 1);
                        }
                        row += 1;
                    }
                }
                myfile.close();

                int classified = 0;
                int truth = 0;

                std::string pNearestNeighbors = kNearestNeighbor(2, currentFeatures);
                std::cout << "KNN says this is the object: " + pNearestNeighbors + "\n"
                          << std::endl;

                classified = getClassified(pNearestNeighbors);

                // ask user what this is, then add tally
                cv::String nameSpecified;
                std::cout << "The true name for current object: ";
                std::cin >> nameSpecified;

                truth = getTruth(nameSpecified);

                // adds to the current matrix and then stores that into a file
                confusionMatrix[classified][truth] += 1;
                std::cout << "The Confusion Matrix:" << std::endl;

                for (int i = 0; i < 10; i++)
                {
                    for (int j = 0; j < 10; j++)
                    {
                        std::cout << confusionMatrix[i][j];
                    }
                    std::cout << std::endl;
                }

                // save in a file
                storeConfusionMatrix(confusionMatrix);
                std::string KNNLabel = "KNN classified as: " + pNearestNeighbors;
                cv::putText(dst, KNNLabel, Point(20, 50), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(54, 117, 136), 2);
            }

            imshow("Display window", dst);
            k = cv::waitKey(0);
        }
    }
    return 0;
}