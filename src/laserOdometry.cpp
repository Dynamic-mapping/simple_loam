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
pcl::PointCloud<PointType>::Ptr targetPoint(new pcl::PointCloud<PointType>());
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

typedef PointType PointT;

class PointToPlaneCostFunction
        : public ceres::SizedCostFunction<1 /* number of residuals */,
                                          6 /* size of first parameter */>
{
public:
    const Eigen::Vector3d& p_src;
    const Eigen::Vector3d& p_nor;
    const double& p_res;
    PointToPlaneCostFunction(const Eigen::Vector3d &src, const Eigen::Vector3d &nor, const double &res) :
            p_src(src), p_nor(nor), p_res(res)
    {}
    virtual ~PointToPlaneCostFunction() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        double p[3] = {p_src[0], p_src[1], p_src[2]};
        double camera[6] = {parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]};
        ceres::AngleAxisRotatePoint(camera, p, p);

        // camera[3,4,5] are the translation.
//        p[0] += camera[3];
//        p[1] += camera[4];
//        p[2] += camera[5];

        // The error is the difference between the predicted and observed position.
        residuals[0] = p_res;

        Eigen::MatrixXd jaconb = Eigen::MatrixXd::Zero(3, 9);
        Eigen::MatrixXd transb = Eigen::MatrixXd::Zero(3, 3);

        // theta x
        float srx = sin(camera[0]);
        float crx = cos(camera[0]);
        // theta y
        float sry = sin(camera[1]);
        float cry = cos(camera[1]);
        // theta z
        float srz = sin(camera[2]);
        float crz = cos(camera[2]);

        // derivative for theta x
        jaconb(0, 0) = 0; jaconb(0, 1) = crz*crx*sry + srx*srz; jaconb(0, 2) = -crz*srx*sry + crx*srz;
        jaconb(0, 3) = 0; jaconb(0, 4) = srz*crx*sry - srx*crz; jaconb(0, 5) = -srz*srx*sry - crx*crz;
        jaconb(0, 6) = 0; jaconb(0, 7) = crx*cry;               jaconb(0, 8) = -srx*cry;

        // derivative for theta y
        jaconb(1, 0) = -sry*crz; jaconb(1, 1) =  crz*srx*cry;   jaconb(1, 2) =  crz*crx*cry;
        jaconb(1, 3) = -sry*srz; jaconb(1, 4) =  srz*srx*cry;   jaconb(1, 5) =  srz*crx*cry;
        jaconb(1, 6) = -cry;     jaconb(1, 7) = -srx*sry;       jaconb(1, 8) = -crx*sry;

        // derivative for theta z
        jaconb(2, 0) = -cry*srz; jaconb(2, 1) = -srz*srx*sry - crx*crz;   jaconb(2, 2) = -srz*crx*sry + srx*crz;
        jaconb(2, 3) =  cry*crz; jaconb(2, 4) =  crz*srx*sry - crx*srz;   jaconb(2, 5) =  crz*crx*sry + srx*srz;
        jaconb(2, 6) =  0;       jaconb(2, 7) =  0;                       jaconb(2, 8) =  0;

        // jaconb for translation
        transb(0, 0) = cry*crz; transb(0, 1) = crz*srx*sry - crx*srz;   transb(2, 2) = crz*crx*sry + srx*srz;
        transb(1, 0) = cry*srz; transb(1, 1) = srz*srx*sry + crx*crz;   transb(2, 2) = srz*crx*sry - srx*crz;
        transb(2, 0) = -sry;    transb(2, 1) = srx*cry;                 transb(2, 2) = crx*cry;


        if (jacobians != NULL && jacobians[0] != NULL) {

            jacobians[0][0] = (jaconb(0, 0)*p[0] + jaconb(0, 1)*p[1] + jaconb(0, 2)*p[2])*p_nor[0] +
                              (jaconb(0, 3)*p[0] + jaconb(0, 4)*p[1] + jaconb(0, 5)*p[2])*p_nor[1] +
                              (jaconb(0, 6)*p[0] + jaconb(0, 7)*p[1] + jaconb(0, 8)*p[2])*p_nor[2];
            jacobians[0][1] = (jaconb(1, 0)*p[0] + jaconb(1, 1)*p[1] + jaconb(1, 2)*p[2])*p_nor[0] +
                              (jaconb(1, 3)*p[0] + jaconb(1, 4)*p[1] + jaconb(1, 5)*p[2])*p_nor[1] +
                              (jaconb(1, 6)*p[0] + jaconb(1, 7)*p[1] + jaconb(1, 8)*p[2])*p_nor[2];
            jacobians[0][2] = (jaconb(2, 0)*p[0] + jaconb(2, 1)*p[1] + jaconb(2, 2)*p[2])*p_nor[0] +
                              (jaconb(2, 3)*p[0] + jaconb(2, 4)*p[1] + jaconb(2, 5)*p[2])*p_nor[1] +
                              (jaconb(2, 6)*p[0] + jaconb(2, 7)*p[1] + jaconb(2, 8)*p[2])*p_nor[2];

            jacobians[0][3] = transb(0, 0) * p_nor[0] + transb(0, 1) * p_nor[1] + transb(0, 2) * p_nor[2];
            jacobians[0][4] = transb(1, 0) * p_nor[0] + transb(1, 1) * p_nor[1] + transb(1, 2) * p_nor[2];
            jacobians[0][5] = transb(2, 0) * p_nor[0] + transb(2, 1) * p_nor[1] + transb(2, 2) * p_nor[2];
        }
        return true;
    }
};

struct PointToPlaneError
{
    const Eigen::Vector3d& p_dst;
    const Eigen::Vector3d& p_src;
    const Eigen::Vector3d& p_nor;

    PointToPlaneError(const Eigen::Vector3d& dst, const Eigen::Vector3d& src, const Eigen::Vector3d& nor) :
    p_dst(dst), p_src(src), p_nor(nor)
    {
    }

    // Factory to hide the construction of the CostFunction object from the client code.

    static ceres::CostFunction* Create(const Eigen::Vector3d& observed, const Eigen::Vector3d& worldPoint, const Eigen::Vector3d& normal)
    {
        return (new ceres::AutoDiffCostFunction<PointToPlaneError, 1, 6>(new PointToPlaneError(observed, worldPoint, normal)));
    }

    template <typename T>
    bool operator()(const T * const camera, T* residuals) const
    {

        T p[3] = {T(p_src[0]), T(p_src[1]), T(p_src[2])};
        ceres::AngleAxisRotatePoint(camera, p, p);

        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // The error is the difference between the predicted and observed position.
        residuals[0] = (p[0] - T(p_dst[0])) * T(p_nor[0]) + \
                       (p[1] - T(p_dst[1])) * T(p_nor[1]) + \
                       (p[2] - T(p_dst[2])) * T(p_nor[2]);

//        std::cout <<"p0 is " << p[0] <<" p1 is " << p[1] << "p2 is " << p[2] << std::endl;
        return true;
    }
};

inline double Square(const PointT point)
{
    return (pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2));
}

inline double pointNorm(const PointT point)
{
    return sqrt(Square(point));
}

inline PointT pointAdd(const PointT &p1, const PointT &p2, double pa1, double pa2)
{
    PointT p_out;
    p_out.x = p1.x * pa1 + p2.x * pa2;
    p_out.y = p1.y * pa1 + p2.y * pa2;
    p_out.z = p1.z * pa1 + p2.z * pa2;
    return p_out;
}

inline PointT pointCross2(const PointT &p1, const PointT &p2)
{
    PointT cross;
    cross.x = p1.y * p2.z - p1.z * p2.y;
    cross.y = p1.z * p2.x - p1.x * p2.z;
    cross.z = p1.x * p2.y - p1.y * p2.x;
    return cross;
}

inline PointT pointCross3(PointT p1, PointT p2, PointT p3)
{
    return pointCross2(pointCross2(p1, p2), p3);
}

inline PointT transPoint(PointT pIn, Eigen::MatrixXd rot, Eigen::VectorXd trans)
{
    PointT pOut;
    Eigen::VectorXd point(3);
    point(0) = pIn.x; point(1) = pIn.y; point(2) = pIn.z;
    point = rot*point + trans;
    pOut.x = point(0); pOut.y = point(1); pOut.z = point(2);
    return pOut;
}

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

ceres::Solver::Options getOptions()
{
    ceres::Solver::Options options;
//    options.update_state_every_iteration = true;
    options.preconditioner_type = ceres::IDENTITY;
    options.linear_solver_type  = ceres::DENSE_QR;
    options.min_trust_region_radius            = 1e-5;
    options.max_num_iterations                 = 3;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    return options;
}

ceres::Solver::Options getOptionsMedium()
{
    ceres::Solver::Options options;
    std::cout << "linear algebra: " << options.sparse_linear_algebra_library_type << std::endl;
    std::cout << "linear solver:  " << options.linear_solver_type << std::endl;

    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY;
    return options;
}

void solve(ceres::Problem &problem, bool smallProblem = true)
{
    ceres::Solver::Summary summary;
    ceres::Solve(smallProblem ? getOptions() : getOptionsMedium(), &problem, &summary);
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
              targetPoint->clear();
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

                  float a012 = pointNorm(pointCross2(pointAdd(pointSel, tripod1, 1, -1),
                                                     pointAdd(pointSel, tripod2, 1, -1)));

                  float l12 = pointNorm(pointAdd(tripod1, tripod2, 1, -1));

                  PointT p_n = pointCross3(pointAdd(tripod1, tripod2, 1, -1),
                                           pointAdd(pointSel, tripod1, 1, -1),
                                           pointAdd(pointSel, tripod2, 1, -1));

                  float ld2 = a012 / l12;

                  //!!!!!!!!!!!!!!!!!!! S is very important
                  float s = 1;
                  if (iterCount >= 5) {
                    s = 1 - 1.8 * fabs(ld2);
                  }
                  coeff.x = s * p_n.x/a012/l12;
                  coeff.y = s * p_n.y/a012/l12;
                  coeff.z = s * p_n.z/a012/l12;
                  coeff.intensity = s * ld2;

                  if (s > 0.1 && ld2 != 0) {
                    laserCloudOri->push_back(cornerPointsSharp->points[i]);
                    coeffSel->push_back(coeff);
                    targetPoint->push_back(tripod1);
                  }
                }
                //dd
              }

              int corner_points = coeffSel->size();
              std::cout << "corner points is " << corner_points << std::endl;
              int pointSelNum = laserCloudOri->points.size();
              if (pointSelNum < 10) {
                continue;
              }

              std::cout << "coeff size is " << laserCloudOri->points.size() << std::endl;
              cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
              cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
              cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
              cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
              cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
              cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
              for (size_t pid = 0 ; pid < laserCloudOri->points.size(); pid++) {

                  Eigen::Vector3d src, nor;
                  src[0] = laserCloudOri->points[pid].x;
                  src[1] = laserCloudOri->points[pid].y;
                  src[2] = laserCloudOri->points[pid].z;

                  nor[0] = coeffSel->points[pid].x;
                  nor[1] = coeffSel->points[pid].y;
                  nor[2] = coeffSel->points[pid].z;

                  matA.at<float>(i, 0) = arx;
                  matA.at<float>(i, 1) = ary;
                  matA.at<float>(i, 2) = arz;
                  matA.at<float>(i, 3) = atx;
                  matA.at<float>(i, 4) = aty;
                  matA.at<float>(i, 5) = atz;
                  matB.at<float>(i, 0) = -0.05 * coeffSel->points[pid].intensity;

//                  ceres::Problem problem;
//                  ceres::CostFunction* cost_function = new PointToPlaneCostFunction(src, nor, res);
//                  problem.AddResidualBlock(cost_function,
//                                           new ceres::HuberLoss(0.5),
//                                           transform);

//                  ceres::CostFunction* cost_function = PointToPlaneError::Create(dst, src, nor);
//                  problem.AddResidualBlock(cost_function,
//                                           NULL,
//                                           transform);

              }

              /*
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

              */


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

		  }

		  

		  // Step 3 output
		  {
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
