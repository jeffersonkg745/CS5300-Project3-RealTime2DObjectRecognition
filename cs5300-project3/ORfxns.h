/**
 * Kaelyn Jefferson
 * CS5300 Project 3
 * Object recognition functions (header file) used for project 3.
 */

#ifndef ORFXNS_H
#define ORFXNS_H

// Threshold algorithm (Q1)
int thresholdAlgo(cv::Mat &src, cv::Mat &dst);

// Cleaning up the binary images (Q2)
int erodePhoto(cv::Mat &src, cv::Mat &dst);
int dilatePhoto(cv::Mat &src, cv::Mat &dst);

// Segmenting the binary images into regions (Q3)
cv::Mat applyConnectedComponentsAnalysis(cv::Mat &src, cv::Mat &dst);

// Compute features for the major region with regionId (Q4)
std::string regionFeatures(cv::Mat &src, cv::Mat &dst, int regionId);

// Collect training data (Q5)
int storeFeatures(cv::String nameGiven, cv::String currentFeatures);

// helper function to store the distance metrics into a file
int storeDistanceMetrics(cv::String featureString, long double distance);

// Classify new images (Q6)
std::string classNewImages(cv::String featureVectorStr);

// a different classifier (Q7)
std::string kNearestNeighbor(int k, cv::String featureVectorStr);

// helper functions for making the confusion matrix (Q8)
int getTruth(std::string nameSpecified);
int getClassified(std::string pNearestNeighbors);

#endif /* ORFXNS_H */
