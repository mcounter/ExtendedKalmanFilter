#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "kalman_filter.h"
#include "tools.h"

// Sensor fusion
class FusionEKF
{

private:

	// Previous timestamp
	long long previous_timestamp_;

public:

	/**
	* Constructor
	*/
	FusionEKF();

	/**
	* Destructor
	*/
	virtual ~FusionEKF();

	/**
	* Run the whole flow of the Kalman Filter from here.
	*/
	void ProcessMeasurement(const MeasurementPackage &measurement_pack);

	/**
	* Kalman Filter process 
	*/
	KalmanFilterProcessLinMov ekf_;
	KalmanFilterMeasurement* measurements_[SensorTypeSize];
};

#endif /* FusionEKF_H_ */
