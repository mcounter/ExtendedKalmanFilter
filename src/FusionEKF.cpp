#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <exception>
#include <stdexcept>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

FusionEKF::FusionEKF()
{
	previous_timestamp_ = 0;
  
	//Process noise vector
	VectorXd q = VectorXd(2);
	q << 9, 9;

	// Initial process covariance vector
	MatrixXd P = MatrixXd(4, 4);
	P <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1000, 0,
		0, 0, 0, 1000;

	ekf_ = KalmanFilterProcessLinMov(q, P);

	// Measurement covariance matrix - laser
	MatrixXd R_laser = MatrixXd(2, 2);
	R_laser <<
		0.0225, 0,
		0, 0.0225;

	// Measurement covariance matrix - radar
	MatrixXd R_radar = MatrixXd(3, 3);
	R_radar <<
		0.09, 0, 0,
		0, 0.0009, 0,
		0, 0, 0.09;

	for (int i = 0; i < SensorTypeSize; i++)
	{
		switch (i)
		{
			case MeasurementPackage::LASER:
				measurements_[i] = new ExtKalmanFilterLidar(ekf_, R_laser);
				break;

			case MeasurementPackage::RADAR:
				measurements_[i] = new ExtKalmanFilterRadar(ekf_, R_radar);
				break;

			default:
				measurements_[i] = 0;
				break;
		}
	}
}

FusionEKF::~FusionEKF() 
{
	for (int i = 0; i < SensorTypeSize; i++)
	{
		if (measurements_[i])
		{
			delete measurements_[i];
		}
	}
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack)
{
	try
	{
		if (!measurements_[measurement_pack.sensor_type_])
		{
			throw std::runtime_error("Unsupported sensor type.");
		}

		if (previous_timestamp_ > 0)
		{
			float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
			measurements_[measurement_pack.sensor_type_]->Predict(dt);
		}

		previous_timestamp_ = measurement_pack.timestamp_;

		measurements_[measurement_pack.sensor_type_]->Update(measurement_pack.raw_measurements_);

		// print the output
		cout << "EKF: " << endl;
		cout << "x_ = " << ekf_.x_ << endl;
		cout << "P_ = " << ekf_.P_ << endl;
	}
	catch (std::exception& ex)
	{
		cout << ex.what() << endl;
	}
}
