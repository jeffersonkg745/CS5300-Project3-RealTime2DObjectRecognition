/**
 * Kaelyn Jefferson
 * CS5300 Project 3
 * Object recognition functions used for project 3.
 */
#include "ORfxns.h"
#include <math.h>
#include <cmath>
#include <numbers>
#include <fstream>
#include <string.h>
#include <cstring>
#include <stdlib.h>
#include <list>
#include <map>
#include <string>
using namespace cv;
using namespace std;
#include <array>

/**
 * @brief Question 1: Threshold algorithm separates the object from the background and produces a binary image.
 *
 * @param src cv::Mat type image
 * @param dst cv::Mat type image
 * @return int
 */
int thresholdAlgo(cv::Mat &src, cv::Mat &dst)
{
    // create dst, convert src from RGB color space to HUV color space
    dst.create(src.size(), src.type());
    cv::Mat hsvVersion;
    cvtColor(src, hsvVersion, cv::COLOR_BGR2HSV);

    // for very saturated pixels, set the pixel "value" in hsv space to be lower
    // in order to visualize diagram here --> https://learn.leighcotnoir.com/artspeak/elements-color/hue-value-saturation/
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            cv::Vec3i currentPixel = {0, 0, 0};

            for (int c = 0; c < 3; c++)
            {
                currentPixel[c] = hsvVersion.at<cv::Vec3b>(i, j)[c];
            }
            if (currentPixel[1] > 200)
            {
                currentPixel[2] -= 100;
            }
            hsvVersion.at<cv::Vec3b>(i, j) = currentPixel;
        }
    }

    // convert the hsv to color space
    cv::Mat rgbVersion;
    rgbVersion.create(src.size(), src.type());
    cvtColor(hsvVersion, rgbVersion, cv::COLOR_HSV2BGR);

    // now we have an image that has a large contrast between white and dark colors
    // set anything close to white to black, otherwise set everything else to white
    for (int i = 0; i < rgbVersion.rows; i++)
    {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < rgbVersion.cols; j++)
        {
            int blue = ptr[j][0];
            int green = ptr[j][1];
            int red = ptr[j][2];

            if (blue > 100 && green > 100 && red > 100)
            {
                cv::Vec3i blackPix = {0, 0, 0};
                dst.at<cv::Vec3b>(i, j) = blackPix;
            }
            else
            {
                cv::Vec3i blackPix = {255, 255, 255};
                dst.at<cv::Vec3b>(i, j) = blackPix;
            }
        }
    }
    return 0;
}

/**
 * @brief Question 2: Cleaning up the binary images, uses openCV function for eroding the photo.
 *
 * @param src cv::Mat type image
 * @param dst cv::Mat type image
 * @return int
 */
int erodePhoto(cv::Mat &src, cv::Mat &dst)
{
    dst.create(src.size(), src.type());
    cv::erode(src, dst, cv::Mat(), cv::Point(-1, -1), 1);
    return 0;
}
/**
 * @brief Question 2: Cleaning up the binary images, uses openCV function for dilating the photo.
 *
 * @param src cv::Mat type image
 * @param dst cv::Mat type image
 * @return int
 */
int dilatePhoto(cv::Mat &src, cv::Mat &dst)
{
    dst.create(src.size(), src.type());
    cv::dilate(src, dst, cv::Mat(), cv::Point(-1, -1), 1);
    return 0;
}

/**
 * @brief Question 3: Segmenting the image into regions using opencv functionality for connected components
 * analysis on the thresholded image to get regions. Function ignores small regions.
 *
 * @param src cv::Mat type image
 * @param dst cv::Mat type image that is the regionMap for the image
 * @return int
 */
cv::Mat applyConnectedComponentsAnalysis(cv::Mat &src, cv::Mat &dst)
{
    // convert src to 1 channel, 8 bit image for the opencv cca function
    cv::Mat binIm;
    cv::cvtColor(src, binIm, cv::COLOR_BGR2GRAY);
    dst.create(src.size(), src.type());

    // OpenCV function for connected components algorithm returns num of regions total
    // referenced code for correct Mat types: https://stackoverflow.com/questions/34321567/opencv3-accessing-label-centroids
    cv::Mat regionMap;
    cv::Mat1i stats;
    cv::Mat1d centroids;
    int connectivity = 4;
    int num = cv::connectedComponentsWithStats(binIm, regionMap, stats, centroids, connectivity, CV_32S, cv::CCL_DEFAULT);

    // colors used for the regions
    cv::Vec3b color1 = {128, 0, 128};  // purple
    cv::Vec3b color2 = {79, 121, 66};  // green
    cv::Vec3b color3 = {52, 80, 92};   // dark blue
    cv::Vec3b color4 = {255, 159, 48}; // orange
    cv::Vec3b color5 = {165, 42, 42};  // red
    cv::Vec3b currentColorToAdd = color1;

    // set the regions to different colors if there are fewer than 5 regions
    // skip 0 which holds background region
    if (num < 5)
    {
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                for (int c = 0; c < 3; c++)
                {
                    // goes through each of the regions coloring them
                    for (int k = 1; k < num; k++)
                    {
                        if (k == 1)
                        {
                            currentColorToAdd = color1;
                        }
                        else if (k == 2)
                        {
                            currentColorToAdd = color2;
                        }
                        else if (k == 3)
                        {
                            currentColorToAdd = color3;
                        }
                        else if (k == 4)
                        {
                            currentColorToAdd = color4;
                        }

                        // checks to see if the pixel is in the centroid area (use stats from cca function)
                        // ignores areas that are too small
                        int minRow = stats(k, cv::CC_STAT_TOP);
                        int maxRow = stats(k, cv::CC_STAT_TOP) + stats(k, cv::CC_STAT_HEIGHT);
                        int minCol = stats(k, cv::CC_STAT_LEFT);
                        int maxCol = stats(k, cv::CC_STAT_LEFT) + stats(k, cv::CC_STAT_WIDTH);
                        int area = stats(k, cv::CC_STAT_AREA);

                        if (regionMap.at<int>(i, j) > 0 && i >= minRow && i < maxRow && j >= minCol && j <= maxCol && area > 150)
                        {
                            dst.at<cv::Vec3b>(i, j)[c] = currentColorToAdd[c];
                        }
                    }
                }
            }
        }
    }

    return regionMap;
}

/**
 * @brief Question 4: computing the features for each major region
 *
 * @param src cv::Mat binary image that holds the regionMap
 * @param dst cv::Mat type image
 * @return int ID of the region
 */
std::string regionFeatures(cv::Mat &regionMap, cv::Mat &dst, int regionId)
{

    // trying the opencv moments function

    float M00 = 0;                    // count of all pixels
    float M10 = 0;                    // sum of x values
    float M01 = 0;                    // sum of y values
    float M11 = 0;                    // for the angle formulas
    float M20 = 0;                    // for the angle formulas
    float M02 = 0;                    // for the angle formulas
    float xprimeMax = 0;              // find the major and minor axes
    float xprimeMin = 100000000;      // find the major and minor axes
    float yprimeMax = 0;              // find the major and minor axes
    float yprimeMin = 100000000;      // find the major and minor axes
    float muPQAxesRelativeMoment = 0; // find the major and minor axes

    // compute the axis of least central moment
    for (int i = 0; i < regionMap.rows; i++)
    {
        for (int j = 0; j < regionMap.cols; j++)
        {
            if (regionMap.at<int>(i, j) == regionId)
            {
                // convert i and j to x and y
                int x = j;
                int y = (regionMap.rows - 1) - i;

                // add to Mpq for each pixel
                M00 += 1;
                M10 += x;
                M01 += y;
            }
        }
    }

    // calculate the centroid (x bar, y bar)
    float xbar = M10 / M00;
    float ybar = M01 / M00;

    // compute the angle
    for (int i = 0; i < regionMap.rows; i++)
    {
        for (int j = 0; j < regionMap.cols; j++)
        {
            if (regionMap.at<int>(i, j) == regionId)
            {
                // convert i and j to x and y
                int x = j;
                int y = (regionMap.rows - 1) - i;

                // add to Mpq for each pixel
                M11 += ((x - xbar) * (y - ybar)); // covariance
                M20 += ((x - xbar) * (x - xbar)); // variance
                M02 += ((y - ybar) * (y - ybar)); // variance
            }
        }
    }

    // tells orientation of major axes (keep in radians bc use cos/tan with it)
    float alpha = 0.5 * atan2(2 * M11, M20 - M02);

    // we reset the M11, M20, M02 to be zero (and calculate it again for rotation, translation invariant qualities)
    M11 = 0;
    M20 = 0;
    M02 = 0;
    float M30 = 0;

    // find the major and minor axes
    for (int i = 0; i < regionMap.rows; i++)
    {
        for (int j = 0; j < regionMap.cols; j++)
        {
            if (regionMap.at<int>(i, j) == regionId)
            {
                // convert i and j to x and y
                int x = j;
                int y = (regionMap.rows - 1) - i;

                // calculate the projected points onto major axes
                //  compute moment relative to that axes
                float xprime = (x - xbar) * cos(alpha) + (y - ybar) * sin(alpha);  // major axes
                float yprime = (x - xbar) * -sin(alpha) + (y - ybar) * cos(alpha); // minor axes

                // set the max and mins here
                if (xprime < xprimeMin)
                {
                    xprimeMin = xprime;
                }
                if (xprime > xprimeMax)
                {
                    xprimeMax = xprime;
                }
                if (yprime < yprimeMin)
                {
                    yprimeMin = yprime;
                }
                if (yprime > yprimeMax)
                {
                    yprimeMax = yprime;
                }

                // set the *relative* moments which are invariant to translation, rotation, scale
                M11 += xprime * yprime;
                M20 += xprime * xprime;
                M02 += yprime * yprime;
                M30 += xprime * xprime * xprime;
            }
        }
    }

    // draw the major axes in x,y coordinates (use Point1 and Point2)
    // note the 200 is a random int to draw a line of some length
    float P1X = xbar;
    float P1Y = ybar;
    float P2X = xbar + 200 * cos(alpha);
    float P2Y = ybar + 200 * sin(alpha);

    // convert back to i,j coordinates in the image and use these to plot the axis of least central moment
    //  https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
    int P1J = P1X;
    int P1I = (regionMap.rows - 1) - P1Y;
    int P2J = P2X;
    int P2I = (regionMap.rows - 1) - P2Y;
    cv::Point point1 = cv::Point(P1J, P1I);
    cv::Point point2 = cv::Point(P2J, P2I);
    cv::line(dst, point1, point2, cv::Scalar(255, 0, 0), 5);

    // plotting the centroid
    int Jcent = xbar;
    int Icent = (regionMap.rows - 1) - ybar;
    cv::Point point = cv::Point(Jcent, Icent);
    cv::circle(dst, point, 5, cv::Scalar(0, 0, 255), 5);

    // make rectangle with just connecting lines
    // first unrotate and add back x bar to get in the x y plane
    int rotXMax = (xprimeMax * cos(alpha)) - (yprimeMax * sin(alpha)) + xbar;
    int rotYMax = (xprimeMax * sin(alpha)) - (yprimeMax * cos(alpha)) + ybar;
    int rotXMin = (xprimeMin * cos(alpha)) - (yprimeMin * sin(alpha)) + xbar;
    int rotYMin = (xprimeMin * sin(alpha)) - (yprimeMin * cos(alpha)) + ybar;

    // convert to i and j
    int JprimeMax = rotXMax;
    int JprimeMin = rotXMin;
    int IprimeMax = (regionMap.rows - 1) - rotYMax;
    int IprimeMin = (regionMap.rows - 1) - rotYMin;

    // drawing the bounding box
    cv::Point c1 = cv::Point(JprimeMax, IprimeMax);
    cv::Point c2 = cv::Point(JprimeMax, IprimeMin);
    cv::Point c3 = cv::Point(JprimeMin, IprimeMax);
    cv::Point c4 = cv::Point(JprimeMin, IprimeMin);
    cv::line(dst, c1, c2, cv::Scalar(255, 0, 0), 5);
    cv::line(dst, c2, c4, cv::Scalar(255, 0, 0), 5);
    cv::line(dst, c4, c3, cv::Scalar(255, 0, 0), 5);
    cv::line(dst, c3, c1, cv::Scalar(255, 0, 0), 5);

    // save the features below to file titled "DBFeatureFile.txt"
    // M11, M20, M02, width/height ratio of the bounding box
    cv::String features;
    features.append(std::to_string(M11));
    features.append(", ");
    features.append(std::to_string(M20));
    features.append(", ");
    features.append(std::to_string(M02));
    features.append(", ");
    // chose a fourth moment for the comparisons: https://docs.opencv.org/3.4/d8/d23/classcv_1_1Moments.html
    features.append(std::to_string(M30));
    features.append(", ");

    return features;
}

/**
 * @brief Question 5: collect training data in the format of "namegiven : features" and store it into a file
 *
 * @param cv::string nameGiven
 * @param std::string featureString
 * @return int
 */
int storeFeatures(cv::String nameGiven, std::string featureString)
{
    // obtained from: https://www.tutorialspoint.com/how-to-append-text-to-a-text-file-in-cplusplus
    // https://www.w3schools.com/cpp/cpp_files.asp
    std::ofstream fout;
    fout.open("DBFeatureFile.txt", std::ios_base::app);
    cv::String s = nameGiven + " , " + featureString;
    fout << s << std::endl;
    std::cout << "Object feature saved." << std::endl;
    fout.close();
    return 0;
}

/**
 * @brief Helper function to store distance metrics in "DistanceMetrics.txt" and used with classNewImages fxn below.
 *
 * @param cv::string featureString
 * @param long double distance
 * @return int
 */
int storeDistanceMetrics(cv::String featureString, long double distance)
{
    std::ofstream fout;
    fout.open("DistanceMetrics.txt", std::ios::app); // appends to the end of the file
    cv::String s = featureString + std::to_string(distance);

    // edge cases that cause string errors
    if (s.length() < 20)
    {
        return 0;
    }
    fout << s << std::endl;
    fout.close();
    return 0;
}

/**
 * @brief Helper function to reset distance metrics in "DistanceMetrics.txt".
 *
 * @return int
 */
int resetDistanceMetrics()
{
    std::ofstream fout;
    fout.open("DistanceMetrics.txt", std::ios::trunc); // rewrites all the distance metrics for each comparison
    fout.close();
    return 0;
}

/**
 * @brief Question 6: classify new images using scaled euclidean distance metric and nearest neighbor recognition
 *
 * @param cv::string featureVector string
 * @return int
 */
std::string classNewImages(cv::String featureVectorStr)
{

    int totalCountFeatures = 0;
    double populationSum[4] = {0, 0, 0, 0}; // need to divide this by totalCountFeatures to find the mean
    // obtained from: https://www.udacity.com/blog/2021/05/how-to-read-from-a-file-in-cpp.html
    std::ifstream myfile;
    myfile.open("DBFeatureFile.txt");
    std::string myline;

    // CURRENT FEATURE VEC: make arr for feature vector of the current photo feature vector
    std::string currentFeatureVector = featureVectorStr;

    std::string currentFeatures[5] = {};
    std::string delim = ", ";
    for (int i = 0; i < 5; i++)
    {
        currentFeatures[i] = currentFeatureVector.substr(0, currentFeatureVector.find(delim));
        currentFeatureVector.erase(0, currentFeatureVector.find(delim) + delim.length());
    }
    myfile.clear();
    myfile.seekg(0);

    // FIRST LOOP: finds the total num of things, the sum of each FeatureVector, and the avg
    if (myfile.is_open())
    {
        while (myfile)
        {
            // https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
            std::getline(myfile, myline);
            std::string s = myline;

            std::string features[5] = {};
            for (int i = 0; i < 5; i++)
            {
                features[i] = s.substr(0, s.find(delim));
                s.erase(0, s.find(delim) + delim.length());
                if (i > 0)
                {
                    populationSum[i - 1] += atof(features[i].c_str());
                }
            }
            totalCountFeatures += 1;
        }
    }

    // resetting to the beginning of the file: https://stackoverflow.com/questions/5343173/returning-to-beginning-of-file-after-getline
    myfile.clear();
    myfile.seekg(0);

    // LOOP 2: calculating the std dev of all the distance metrics (11, 20, 02, 30)
    double sumNum[5] = {0, 0, 0, 0, 0};
    if (myfile.is_open())
    {
        while (myfile)
        {
            std::getline(myfile, myline);
            std::string s = myline;
            std::string features[5] = {};
            for (int i = 0; i < 5; i++)
            {
                features[i] = s.substr(0, s.find(delim));
                if (i > 0) // skip zero bc it is the title of the object
                {
                    double x = atof(features[i].c_str());
                    double mean = populationSum[i] / totalCountFeatures;

                    double num = (x - mean) * (x - mean);
                    sumNum[i - 1] += num;
                }
                s.erase(0, s.find(delim) + delim.length());
            }
        }
    }
    myfile.clear();
    myfile.seekg(0);

    double stddev[5] = {};

    stddev[0] = sqrt(sumNum[0] / totalCountFeatures);
    stddev[1] = sqrt(sumNum[1] / totalCountFeatures);
    stddev[2] = sqrt(sumNum[2] / totalCountFeatures);
    stddev[3] = sqrt(sumNum[3] / totalCountFeatures);

    // LOOP 3: go through the file again and compare the current object's features to each of the other ones
    long double closestFeature = 100;
    std::string closestFeatureLine = "";

    if (myfile.is_open())
    {
        while (myfile)
        {
            std::getline(myfile, myline);
            std::string s = myline;
            std::string features[5] = {};
            for (int i = 0; i < 5; i++)
            {
                features[i] = s.substr(0, s.find(delim));

                // avoids label problem with term 0
                if (i > 0)
                {
                    // take the calculation here then add it to the counts above
                    long double x = (atof(currentFeatures[i - 1].c_str()) - atof(features[i].c_str())) / stddev[i - 1];
                    long double EDistMet = x * x;

                    // store all the euclidian distance results in a file to sort later
                    storeDistanceMetrics(myline, EDistMet);

                    if (EDistMet < closestFeature)
                    {
                        closestFeature = EDistMet;
                        closestFeatureLine = myline;
                    }
                }
                s.erase(0, s.find(delim) + delim.length());
            }
        }
    }

    std::cout << "stored all the metrics." << std::endl;

    // get the label for the object to what its closest to
    std::string objectLabel = closestFeatureLine.substr(0, closestFeatureLine.find_first_of(","));

    return objectLabel;
}

// find closest k neighbors in each class and sum the distances
// use Scaled SSD to find multiple neighbors
/**
 * @brief Question 7: classify new images using KNN distance metric and returns the closest category as a string
 *
 * @param cv::string featureVector string
 * @param int k number of photos to compare for each category
 * @return std::string
 */
std::string kNearestNeighbor(int k, cv::String featureVectorStr)
{

    std::map<std::string, std::vector<std::string>> vecmap;

    // get the Name : dist met from the string
    // https://www.geeksforgeeks.org/how-to-insert-data-in-the-map-of-strings/
    std::ifstream myfile;
    myfile.open("DistanceMetrics.txt");
    std::string myline;
    if (myfile.is_open())
    {
        while (myfile)
        {
            std::getline(myfile, myline);
            std::string s = myline;
            if (s.length() < 10)
            {
                break;
            }
            std::string category = s.substr(0, s.find_first_of(","));
            std::string distMetString = s.substr(s.find_last_of(", "), s.length());

            if (atof(distMetString.c_str()) == 0)
            {
                continue;
            }

            // add to the map --> name : distances, update if found
            if (vecmap.find(category) != vecmap.end())
            {
                auto cat = vecmap.find(category);
                std::vector<std::string> currentVec = cat->second;
                currentVec.push_back(distMetString);
                cat->second = currentVec;
            }
            else
            {
                std::vector<std::string> v = {distMetString};
                vecmap.insert(std::pair<std::string, std::vector<std::string>>(category, v));
            }
        }
    }

    double smallest = 100;
    std::string smallestCategory = "";
    // go through map once and sort each list
    for (auto const &pair : vecmap)
    {
        std::string categoryName = pair.first;
        std::vector<std::string> vpair = pair.second;
        std::sort(vpair.begin(), vpair.end());

        double firstNum = -1;
        double secondNum = -1;

        for (std::string x : vpair)
        {
            if (firstNum == -1 && secondNum == -1)
            {
                firstNum = atof(x.c_str());
            }
            else if (firstNum != -1 && secondNum == -1)
            {
                secondNum = atof(x.c_str());
            }
        }
        double sum = firstNum + secondNum;
        if (sum < smallest)
        {
            smallest = sum;
            smallestCategory = categoryName;
        }
    }
    return smallestCategory;
}

/**
 * @brief Helper function to store the matrix in the file called "ConfusionMatrix.txt"
 *
 * @return void
 */
void storeConfusionMatrix(int matrix[10][10])
{
    std::ofstream fout;
    fout.open("ConfusionMatrix.txt", std::ios::trunc);

    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            fout << matrix[i][j];
        }
        fout << std::endl;
    }

    fout.close();
}

/**
 * @brief Helper function to classify the name given by the UI into the categories of images for the
 * truth value of the Confusion matrix
 *
 * @return void
 */
int getTruth(std::string nameSpecified)
{
    int truth = 0;
    if (nameSpecified == "bird")
    {
        truth = 0;
    }
    else if (nameSpecified == "bronzer")
    {
        truth = 1;
    }
    else if (nameSpecified == "butterfly")
    {
        truth = 2;
    }
    else if (nameSpecified == "deer")
    {
        truth = 3;
    }
    else if (nameSpecified == "hairtie")
    {
        truth = 4;
    }
    else if (nameSpecified == "lipstick")
    {
        truth = 5;
    }
    else if (nameSpecified == "remote")
    {
        truth = 6;
    }
    else if (nameSpecified == "sticky")
    {
        truth = 7;
    }
    else if (nameSpecified == "whiteout")
    {
        truth = 8;
    }
    else if (nameSpecified == "wrench")
    {
        truth = 9;
    }

    return truth;
}

/**
 * @brief Helper function to classify the name given by the UI into the categories of images for the
 * classification value of the Confusion matrix
 *
 * @return void
 */
int getClassified(std::string classedAs)
{

    int classified;

    if (classedAs.compare("bird ") == 0)
    {
        classified = 0;
    }
    else if (classedAs.compare("bronzer ") == 0)
    {
        classified = 1;
    }
    else if (classedAs.compare("butterfly ") == 0)
    {
        classified = 2;
    }
    else if (classedAs.compare("deer ") == 0)
    {
        classified = 3;
    }
    else if (classedAs.compare("hairtie ") == 0)
    {
        classified = 4;
    }
    else if (classedAs.compare("lipstick ") == 0)
    {
        classified = 5;
    }
    else if (classedAs.compare("remote ") == 0)
    {
        classified = 6;
    }
    else if (classedAs.compare("sticky ") == 0)
    {
        classified = 7;
    }
    else if (classedAs.compare("whiteout ") == 0)
    {
        classified = 8;
    }
    else if (classedAs.compare("wrench ") == 0)
    {
        classified = 9;
    }

    return classified;
}