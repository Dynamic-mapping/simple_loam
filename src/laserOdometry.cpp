// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

#include <cmath>

#include <loam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

// CERES LIB
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include <ceres/autodiff_cost_function.h>
#include <ceres/autodiff_local_parameterization.h>
#include <ceres/types.h>
#include <ceres/rotation.h>

// EIGEN LIB
#include <Eigen/Dense>
#include <Eigen/LU>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

const float scanPeriod = 0.1;

const int skipFrameNum = 1;
bool systemInited = false;

double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;
double timeImuTrans = 0;

bool newCornerPointsSharp = false;
bool newCornerPointsLessSharp = false;
bool newSurfPointsFlat = false;
bool newSurfPointsLessFlat = false;
bool newLaserCloudFullRes = false;
bool newImuTrans = false;

pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudOri(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr coeffSel(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<pcl::PointXYZ>::Ptr imuTrans(new pcl::PointCloud<pcl::PointXYZ>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<PointType>());
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<PointType>());

int laserCloudCornerLastNum;
int laserCloudSurfLastNum;

int pointSelCornerInd[40000];
float pointSearchCornerInd1[40000];
float pointSearchCornerInd2[40000];

int pointSelSurfInd[40000];
float pointSearchSurfInd1[40000];
float pointSearchSurfInd2[40000];
float pointSearchSurfInd3[40000];

double transform[6] = {0};
double transformSum[6] = {0};

float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
float imuRollLast = 0, imuPitchLast = 0, imuYawLast = 0;
float imuShiftFromStartX = 0, imuShiftFromStartY = 0, imuShiftFromStartZ = 0;
float imuVeloFromStartX = 0, imuVeloFromStartY = 0, imuVeloFromStartZ = 0;

template <typename T>
ceres::MatrixAdapter<T, 1, 4> ColumnMajorAdapter4x3(T* pointer)
{
    return ceres::MatrixAdapter<T, 1, 4>(pointer);
}

void isoToAngleAxis(const Eigen::Isometry3d &pose, double *cam)
{
    ceres::RotationMatrixToAngleAxis(ColumnMajorAdapter4x3(pose.linear().data()), cam);
    Eigen::Vector3d t(pose.translation());
    cam[3] = t.x();
    cam[4] = t.y();
    cam[5] = t.z();
}

Eigen::Isometry3d axisAngleToIso(const double* transform)
{
    Eigen::Isometry3d poseFinal = Eigen::Isometry3d::Identity();
    Eigen::Matrix3d rot;
    ceres::AngleAxisToRotationMatrix(transform, rot.data());
    poseFinal.linear() = rot;
    poseFinal.translation() = Eigen::Vector3d(transform[3], transform[4], transform[5]);
    return poseFinal;
}

void TransformToStart(PointType const * const pi, PointType * const po)
{

  float rx = transform[0];
  float ry = transform[1];
  float rz = transform[2];
  float tx = transform[3];
  float ty = transform[4];
  float tz = transform[5];

  float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
  float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
  float z1 = (pi->z - tz);

  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;

  po->x = cos(ry) * x2 - sin(ry) * z2;
  po->y = y2;
  po->z = sin(ry) * x2 + cos(ry) * z2;
  po->intensity = pi->intensity;
}

void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsSharp2)
{
  timeCornerPointsSharp = cornerPointsSharp2->header.stamp.toSec();

  cornerPointsSharp->clear();
  pcl::fromROSMsg(*cornerPointsSharp2, *cornerPointsSharp);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cornerPointsSharp,*cornerPointsSharp, indices);
  newCornerPointsSharp = true;
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsLessSharp2)
{
  timeCornerPointsLessSharp = cornerPointsLessSharp2->header.stamp.toSec();

  cornerPointsLessSharp->clear();
  pcl::fromROSMsg(*cornerPointsLessSharp2, *cornerPointsLessSharp);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cornerPointsLessSharp,*cornerPointsLessSharp, indices);
  newCornerPointsLessSharp = true;
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsFlat2)
{
  timeSurfPointsFlat = surfPointsFlat2->header.stamp.toSec();

  surfPointsFlat->clear();
  pcl::fromROSMsg(*surfPointsFlat2, *surfPointsFlat);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*surfPointsFlat,*surfPointsFlat, indices);
  newSurfPointsFlat = true;
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsLessFlat2)
{
  timeSurfPointsLessFlat = surfPointsLessFlat2->header.stamp.toSec();

  surfPointsLessFlat->clear();
  pcl::fromROSMsg(*surfPointsLessFlat2, *surfPointsLessFlat);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*surfPointsLessFlat,*surfPointsLessFlat, indices);
  newSurfPointsLessFlat = true;
}

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudFullRes2)
{
  timeLaserCloudFullRes = laserCloudFullRes2->header.stamp.toSec();

  laserCloudFullRes->clear();
  pcl::fromROSMsg(*laserCloudFullRes2, *laserCloudFullRes);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*laserCloudFullRes,*laserCloudFullRes, indices);
  newLaserCloudFullRes = true;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;

    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>
                                         ("/laser_cloud_sharp", 2, laserCloudSharpHandler);

    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>
                                             ("/laser_cloud_less_sharp", 2, laserCloudLessSharpHandler);

    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>
                                      ("/laser_cloud_flat", 2, laserCloudFlatHandler);

    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>
                                          ("/laser_cloud_less_flat", 2, laserCloudLessFlatHandler);

    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>
                                         ("/velodyne_cloud_2", 2, laserCloudFullResHandler);

    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>
                                           ("/laser_cloud_corner_last", 2);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>
                                         ("/laser_cloud_surf_last", 2);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>
                                        ("/velodyne_cloud_3", 2);

    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 5);
    nav_msgs::Odometry laserOdometry;
    laserOdometry.header.frame_id = "/camera_init";
    laserOdometry.child_frame_id = "/laser_odom";

    tf::TransformBroadcaster tfBroadcaster;
    tf::StampedTransform laserOdometryTrans;
    laserOdometryTrans.frame_id_ = "/camera_init";
    laserOdometryTrans.child_frame_id_ = "/laser_odom";

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    PointType pointOri, pointSel, tripod1, tripod2, tripod3, pointProj, coeff;

    int frameCount = skipFrameNum;
    ros::Rate rate(100);
    bool status = ros::ok();
    while (status) {
        ros::spinOnce();

        if (newCornerPointsSharp && newCornerPointsLessSharp && newSurfPointsFlat &&
            newSurfPointsLessFlat && newLaserCloudFullRes &&
            fabs(timeCornerPointsSharp - timeSurfPointsLessFlat) < 0.005 &&
            fabs(timeCornerPointsLessSharp - timeSurfPointsLessFlat) < 0.005 &&
            fabs(timeSurfPointsFlat - timeSurfPointsLessFlat) < 0.005 &&
            fabs(timeLaserCloudFullRes - timeSurfPointsLessFlat) < 0.005) {
          newCornerPointsSharp = false;
          newCornerPointsLessSharp = false;
          newSurfPointsFlat = false;
          newSurfPointsLessFlat = false;
          newLaserCloudFullRes = false;

          if (!systemInited) {
            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            cornerPointsLessSharp = laserCloudCornerLast;
            laserCloudCornerLast = laserCloudTemp;
    
            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;
            laserCloudSurfLast = laserCloudTemp;
    
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
    
            sensor_msgs::PointCloud2 laserCloudCornerLast2;
            pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
            laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            laserCloudCornerLast2.header.frame_id = "/camera";
            pubLaserCloudCornerLast.publish(laserCloudCornerLast2);
    
            sensor_msgs::PointCloud2 laserCloudSurfLast2;
            pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
            laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            laserCloudSurfLast2.header.frame_id = "/camera";
            pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
    
            transformSum[0] += imuPitchStart;
            transformSum[2] += imuRollStart;
    
            systemInited = true;
            continue;
          }









          // Step 1 Caculate the transformation based on the LK gredicent method.
          if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*cornerPointsSharp,*cornerPointsSharp, indices);
            int cornerPointsSharpNum = cornerPointsSharp->points.size();

            // iteration for 25 times
            for (int iterCount = 0; iterCount < 25; iterCount++) {

              laserCloudOri->clear();
              coeffSel->clear();


              //! Caculate the distance for Corner Sharp Points
              for (int i = 0; i < cornerPointsSharpNum; i++) {
                TransformToStart(&cornerPointsSharp->points[i], &pointSel);
                if (iterCount % 5 == 0) {

                  std::vector<int> indices;
                  pcl::removeNaNFromPointCloud(*laserCloudCornerLast,*laserCloudCornerLast, indices);
                  kdtreeCornerLast->nearestKSearch(pointSel, 2, pointSearchInd, pointSearchSqDis);
                  int closestPointInd = -1, minPointInd2 = -1;
                  if (pointSearchSqDis[0] < 25) {
                    closestPointInd = pointSearchInd[0];
                    minPointInd2    = pointSearchInd[1];
                  }

                  pointSearchCornerInd1[i] = closestPointInd;
                  pointSearchCornerInd2[i] = minPointInd2;
                }

                if (pointSearchCornerInd2[i] >= 0) {
                  tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
                  tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]];

                  float x0 = pointSel.x;
                  float y0 = pointSel.y;
                  float z0 = pointSel.z;
                  float x1 = tripod1.x;
                  float y1 = tripod1.y;
                  float z1 = tripod1.z;
                  float x2 = tripod2.x;
                  float y2 = tripod2.y;
                  float z2 = tripod2.z;

                  float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                             * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                             + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                             * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                             + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                             * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                  float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                  float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                           + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                  float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                           - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                  float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                           + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                  float ld2 = a012 / l12;

                  pointProj = pointSel;
                  pointProj.x -= la * ld2;
                  pointProj.y -= lb * ld2;
                  pointProj.z -= lc * ld2;

                  //!!!!!!!!!!!!!!!!!!! S is very important
                  float s = 1;
                  if (iterCount >= 5) {
                    s = 1 - 1.8 * fabs(ld2);
                  }

//                  s = 1;
                  coeff.x = s * la;
                  coeff.y = s * lb;
                  coeff.z = s * lc;
                  coeff.intensity = s * ld2;

                  if (s > 0.1 && ld2 != 0) {
                    laserCloudOri->push_back(cornerPointsSharp->points[i]);
                    coeffSel->push_back(coeff);
                  }
                }
              }

              int corner_points = coeffSel->size();
              std::cout << "corner points is " << corner_points << std::endl;
              int pointSelNum = laserCloudOri->points.size();
              if (pointSelNum < 10) {
                continue;
              }

              cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
              cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
              cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
              cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
              cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
              cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
              for (int i = 0; i < pointSelNum; i++) {
                pointOri = laserCloudOri->points[i];
                coeff = coeffSel->points[i];

                float s = 1;

                float srx = sin(s * transform[0]);
                float crx = cos(s * transform[0]);
                float sry = sin(s * transform[1]);
                float cry = cos(s * transform[1]);
                float srz = sin(s * transform[2]);
                float crz = cos(s * transform[2]);
                float tx = s * transform[3];
                float ty = s * transform[4];
                float tz = s * transform[5];

                float arx = (-s*crx*sry*srz*pointOri.x + s*crx*crz*sry*pointOri.y + s*srx*sry*pointOri.z
                          + s*tx*crx*sry*srz - s*ty*crx*crz*sry - s*tz*srx*sry) * coeff.x
                          + (s*srx*srz*pointOri.x - s*crz*srx*pointOri.y + s*crx*pointOri.z
                          + s*ty*crz*srx - s*tz*crx - s*tx*srx*srz) * coeff.y
                          + (s*crx*cry*srz*pointOri.x - s*crx*cry*crz*pointOri.y - s*cry*srx*pointOri.z
                          + s*tz*cry*srx + s*ty*crx*cry*crz - s*tx*crx*cry*srz) * coeff.z;

                float ary = ((-s*crz*sry - s*cry*srx*srz)*pointOri.x
                          + (s*cry*crz*srx - s*sry*srz)*pointOri.y - s*crx*cry*pointOri.z
                          + tx*(s*crz*sry + s*cry*srx*srz) + ty*(s*sry*srz - s*cry*crz*srx)
                          + s*tz*crx*cry) * coeff.x
                          + ((s*cry*crz - s*srx*sry*srz)*pointOri.x
                          + (s*cry*srz + s*crz*srx*sry)*pointOri.y - s*crx*sry*pointOri.z
                          + s*tz*crx*sry - ty*(s*cry*srz + s*crz*srx*sry)
                          - tx*(s*cry*crz - s*srx*sry*srz)) * coeff.z;

                float arz = ((-s*cry*srz - s*crz*srx*sry)*pointOri.x + (s*cry*crz - s*srx*sry*srz)*pointOri.y
                          + tx*(s*cry*srz + s*crz*srx*sry) - ty*(s*cry*crz - s*srx*sry*srz)) * coeff.x
                          + (-s*crx*crz*pointOri.x - s*crx*srz*pointOri.y
                          + s*ty*crx*srz + s*tx*crx*crz) * coeff.y
                          + ((s*cry*crz*srx - s*sry*srz)*pointOri.x + (s*crz*sry + s*cry*srx*srz)*pointOri.y
                          + tx*(s*sry*srz - s*cry*crz*srx) - ty*(s*crz*sry + s*cry*srx*srz)) * coeff.z;

                float atx = -s*(cry*crz - srx*sry*srz) * coeff.x + s*crx*srz * coeff.y
                          - s*(crz*sry + cry*srx*srz) * coeff.z;

                float aty = -s*(cry*srz + crz*srx*sry) * coeff.x - s*crx*crz * coeff.y
                          - s*(sry*srz - cry*crz*srx) * coeff.z;

                float atz = s*crx*sry * coeff.x - s*srx * coeff.y - s*crx*cry * coeff.z;

                float d2 = coeff.intensity;

                matA.at<float>(i, 0) = arx;
                matA.at<float>(i, 1) = ary;
                matA.at<float>(i, 2) = arz;
                matA.at<float>(i, 3) = atx;
                matA.at<float>(i, 4) = aty;
                matA.at<float>(i, 5) = atz;
                matB.at<float>(i, 0) = -0.05 * d2;
              }
              cv::transpose(matA, matAt);
              matAtA = matAt * matA;
              matAtB = matAt * matB;
              cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

              transform[0] += matX.at<float>(0, 0);
              transform[1] += matX.at<float>(1, 0);
              transform[2] += matX.at<float>(2, 0);
              transform[3] += matX.at<float>(3, 0);
              transform[4] += matX.at<float>(4, 0);
              transform[5] += matX.at<float>(5, 0);

              for(int i=0; i<6; i++){
                if(isnan(transform[i]))
                  transform[i]=0;
              }
              float deltaR = sqrt(
                                  pow(rad2deg(matX.at<float>(0, 0)), 2) +
                                  pow(rad2deg(matX.at<float>(1, 0)), 2) +
                                  pow(rad2deg(matX.at<float>(2, 0)), 2));
              float deltaT = sqrt(
                                  pow(matX.at<float>(3, 0) * 100, 2) +
                                  pow(matX.at<float>(4, 0) * 100, 2) +
                                  pow(matX.at<float>(5, 0) * 100, 2));

              if (deltaR < 0.1 && deltaT < 0.1) {
                break;
              }
            }
          }


          // Step 2 based on the Estimated Translation, transforma the Surf and Edge Pointcloud, also publish the tf
          {
              /// the transform matrix is T_k_k-1
              /// so to compute the transform we need to take the inverse().
              transform[1] = 1.05*transform[1];
              Eigen::Isometry3d trans_sum = axisAngleToIso(transformSum);
              Eigen::Isometry3d trans_tmp = axisAngleToIso(transform);
              trans_sum.matrix() = trans_sum.matrix() * trans_tmp.matrix().inverse();
              isoToAngleAxis(trans_sum, transformSum);

              geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(transformSum[2],
                      -transformSum[0], -transformSum[1]);

              laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
              laserOdometry.pose.pose.orientation.x = -geoQuat.y;
              laserOdometry.pose.pose.orientation.y = -geoQuat.z;
              laserOdometry.pose.pose.orientation.z = geoQuat.x;
              laserOdometry.pose.pose.orientation.w = geoQuat.w;
              laserOdometry.pose.pose.position.x = transformSum[3];
              laserOdometry.pose.pose.position.y = transformSum[4];
              laserOdometry.pose.pose.position.z = transformSum[5];
              pubLaserOdometry.publish(laserOdometry);

              laserOdometryTrans.stamp_ = ros::Time().fromSec(timeSurfPointsLessFlat);
              laserOdometryTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
              laserOdometryTrans.setOrigin(tf::Vector3(transformSum[3],transformSum[4],transformSum[5]));
              tfBroadcaster.sendTransform(laserOdometryTrans);

              frameCount++;

              //! The Current Corner Pointcloud and Edge Pointcloud is the last CornerLess and SurfLess Pointcloud.
              pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
              cornerPointsLessSharp = laserCloudCornerLast;
              laserCloudCornerLast = laserCloudTemp;

              laserCloudTemp = surfPointsLessFlat;
              surfPointsLessFlat = laserCloudSurfLast;
              laserCloudSurfLast = laserCloudTemp;

              laserCloudCornerLastNum = laserCloudCornerLast->points.size();
              laserCloudSurfLastNum = laserCloudSurfLast->points.size();
              if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {
                kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
                kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
              }

              if (frameCount >= skipFrameNum + 1) {
                frameCount = 0;

                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudCornerLast2.header.frame_id = "/camera";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudSurfLast2.header.frame_id = "/camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
              }
            }
        }
        status = ros::ok();
        rate.sleep();
      }

  return 0;
}
