#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

#include "Eigen/Dense"
#include <math.h>

// Note: For maximal compatibility with original main.cpp matrices was not replaced by pointers.
// It caused some extra data copying, which assumed as OK for current implementation.

// Base Extended Kalman filter process
class KalmanFilterProcess
{

private:

	/**
	* Perform matrices validation
	* @param P Initial state covariance
	* @param F Transition matrix
	* @param Q Process covariance matrix
	*/
	void validateMatrices(Eigen::MatrixXd &P, Eigen::MatrixXd &F, Eigen::MatrixXd &Q);

protected:
	// Is process initialized
	bool isProcessInitialized = false;

	/**
	* Calculate transition matrix F (or jacobian)
	* @param dt Time between k and k+1 step
	* @return Transition matrix (or jacobian)
	*/
	virtual Eigen::MatrixXd calcTransitionMatrix(const float &dt);

	/**
	* Calculate transition covariance matrix
	* @param dt Time between k and k+1 step
	* @return New covariance matrix
	*/
	virtual Eigen::MatrixXd calcTransitionCov(const float &dt);

	/**
	* Execute transition function
	* @param dt Time between k and k+1 step
	* @return Vector with new state
	*/
	virtual Eigen::VectorXd performTransition(const float &dt);

public:

	// Is state initialized
	bool initialized_ = false;

	// State vector
	Eigen::VectorXd x_;

	// State covariance matrix
	Eigen::MatrixXd P_;

	// State transition matrix
	Eigen::MatrixXd F_;

	// Process covariance matrix
	Eigen::MatrixXd Q_;

	/**
	* Constructor
	*/
	KalmanFilterProcess();

	/**
	* Constructor with matrices
	* @param P Initial state covariance
	* @param F Transition matrix
	* @param Q Process covariance matrix
	*/
	KalmanFilterProcess(Eigen::MatrixXd &P, Eigen::MatrixXd &F, Eigen::MatrixXd &Q);

	/**
	* Destructor
	*/
	virtual ~KalmanFilterProcess();

	/**
	* Initialize Kalman filter process with matrices
	* This also mark Kalman filter process as initialized, but reset internal state to undefined
	* Internal state must be initialized separately with UpdateState() call
	* @param P Initial state covariance matrix
	* @param F Transition matrix
	* @param Q Process covariance matrix
	*/
	virtual void Init(Eigen::MatrixXd &P, Eigen::MatrixXd &F, Eigen::MatrixXd &Q);

	/**
	* Update (or initialize) Kalman filter process state
	* Kalman filter process must be initialized first with Init() call
	* @param x New state vector
	*/
	virtual void UpdateState(Eigen::VectorXd &x);

	/**
	* Update (or initialize) Kalman filter process state and covariance matrix
	* Kalman filter process must be initialized first with Init() call
	* @param x New state vector
	*/
	virtual void UpdateState(Eigen::VectorXd &x, Eigen::MatrixXd &P);

	/**
	* Predicts the state and the state covariance using the process model
	* @param dt Time between k and k+1 step
	*/
	virtual void Predict(const float &dt);
};

// Extended Kalman filter process for linearly moved model
class KalmanFilterProcessLinMov : public KalmanFilterProcess
{

private:

	/**
	* Perform matrices validation
	* @param q Noise vector
	* @param P Initial state covariance matrix
	*/
	void validateMatrices(Eigen::VectorXd &q, Eigen::MatrixXd &P);

protected:

	/**
	* Calculate transition matrix F (or jacobian)
	* @param dt Time between k and k+1 step
	* @return Transition matrix (or jacobian)
	*/
	Eigen::MatrixXd calcTransitionMatrix(const float &dt);

	/**
	* Calculate transition covariance matrix
	* @param dt Time between k and k+1 step
	* @return New covariance matrix
	*/
	Eigen::MatrixXd calcTransitionCov(const float &dt);

public:

	// noise vector
	Eigen::VectorXd q_;

	/**
	* Constructor
	*/
	KalmanFilterProcessLinMov();

	/**
	* Constructor with matrices
	* @param q Noise vector
	* @param P Initial state covariance matrix
	*/
	KalmanFilterProcessLinMov(Eigen::VectorXd &q, Eigen::MatrixXd &P);

	/**
	* Destructor
	*/
	virtual ~KalmanFilterProcessLinMov();

	/**
	* Initialize Kalman filter process with matrices
	* This also mark Kalman filter process as initialized, but reset internal state to undefined
	* Internal state must be initialized separately with UpdateState() call
	* @param P Initial state covariance matrix
	* @param F Transition matrix
	* @param Q Process covariance matrix
	*/
	void Init(Eigen::MatrixXd &P, Eigen::MatrixXd &F, Eigen::MatrixXd &Q);

	/**
	* Initialize Kalman filter process with matrices
	* This also mark Kalman filter process as initialized, but reset internal state to undefined
	* Internal state must be initialized separately with UpdateState() call
	* @param q Noise vector
	* @param P Initial state covariance matrix
	*/
	virtual void Init(Eigen::VectorXd &q, Eigen::MatrixXd &P);
};

// Base Extended Kalman filter measurement
class KalmanFilterMeasurement
{

private:

	/**
	* Perform matrices validation
	* @param H Measurement matrix
	* @param R Measurement covariance matrix
	*/
	void validateMatrices(Eigen::MatrixXd &H, Eigen::MatrixXd &R);

protected:

	// Is measurement initialized
	bool isMeasurementInitialized = false;

	// Pointer to Extended Kalman filter process
	KalmanFilterProcess* process_;

	/**
	* Calculate measurement matrix H (or jacobian)
	* @return Measurement matrix (or jacobian)
	*/
	virtual Eigen::MatrixXd calcMeasurementMatrix();

	/**
	* Calculate measurement covariance matrix
	* @return New covariance matrix
	*/
	virtual Eigen::MatrixXd calcMeasurementCov();

	/**
	* Predict measurement by state vector
	*/
	virtual Eigen::VectorXd predictMeasurement();

	/**
	* Initialize state from measurement
	* @param z The measurement
	*/
	virtual Eigen::VectorXd initFromMeasurement(const Eigen::VectorXd &z);

	/**
	* Normalize measurement if necessary
	* @param z The measurement
	*/
	virtual Eigen::VectorXd normalizeMeasurement(const Eigen::VectorXd &z);

	/**
	* Calc difference between measurement and predicted value
	* @param z The measurement
	* @param predicted Predicted vector
	*/
	virtual Eigen::VectorXd calcMeasurementDiff(const Eigen::VectorXd &z, const Eigen::VectorXd &predicted);

public:

	// Measurement matrix
	Eigen::MatrixXd H_;

	// Measurement covariance matrix
	Eigen::MatrixXd R_;

	/**
	* Constructor
	* @param process Kalman filter process
	*/
	KalmanFilterMeasurement(KalmanFilterProcess &process);

	/**
	* Constructor with matrices
	* @param process Kalman filter process
	* @param H Measurement matrix
	* @param R Measurement covariance matrix
	*/
	KalmanFilterMeasurement(KalmanFilterProcess &process, Eigen::MatrixXd &H, Eigen::MatrixXd &R);

	/**
	* Destructor
	*/
	virtual ~KalmanFilterMeasurement();

	/**
	* Initialize Kalman filter measurement with matrices
	* This also mark Kalman filter measurement as initialized
	* @param H Measurement matrix
	* @param R Measurement covariance matrix
	*/
	virtual void Init(Eigen::MatrixXd &H, Eigen::MatrixXd &R);

	/**
	* Predicts the state and the state covariance using the process model
	* @param dt Time between k and k+1 step
	*/
	virtual void Predict(const float &dt);

	/**
	* Updates the state after new measurement
	* @param z The measurement at k+1
	*/
	virtual void Update(const Eigen::VectorXd &z);
};

// Extended Kalman filter measurement for LIDAR
class ExtKalmanFilterLidar : public KalmanFilterMeasurement
{

private:

	/**
	* Perform matrices validation
	* @param R Measurement covariance matrix
	*/
	void validateMatrices(Eigen::MatrixXd &R);

protected:

	/**
	* Initialize state from measurement
	* @param z The measurement
	*/
	Eigen::VectorXd initFromMeasurement(const Eigen::VectorXd &z);

public:

	/**
	* Constructor
	* @param process Kalman filter process
	*/
	ExtKalmanFilterLidar(KalmanFilterProcess &process);

	/**
	* Constructor with matrices
	* @param process Kalman filter process
	* @param R Measurement covariance matrix
	*/
	ExtKalmanFilterLidar(KalmanFilterProcess &process, Eigen::MatrixXd &R);

	/**
	* Destructor
	*/
	virtual ~ExtKalmanFilterLidar();

	/**
	* Initialize Kalman filter measurement with matrices
	* This also mark Kalman filter measurement as initialized
	* @param H Measurement matrix
	* @param R Measurement covariance matrix
	*/
	void Init(Eigen::MatrixXd &H, Eigen::MatrixXd &R);

	/**
	* Initialize Kalman filter measurement with matrices
	* This also mark Kalman filter measurement as initialized
	* @param R Measurement covariance matrix
	*/
	virtual void Init(Eigen::MatrixXd &R);
};

// Extended Kalman filter measurement for RADAR
class ExtKalmanFilterRadar : public KalmanFilterMeasurement
{

private:

	/**
	* Perform matrices validation
	* @param R Measurement covariance matrix
	*/
	void validateMatrices(Eigen::MatrixXd &R);

protected:

	// Minimal distance to point of origin
	// Measurement near point of origin has huge measurement error for radial measurement and must be prevented
	float min_distance_ = 0.01;

	/**
	* Calculate measurement matrix H (or jacobian)
	* @return Measurement matrix (or jacobian)
	*/
	Eigen::MatrixXd calcMeasurementMatrix();

	/**
	* Predict measurement by state vector
	*/
	Eigen::VectorXd predictMeasurement();

	/**
	* Initialize state from measurement
	* @param z The measurement
	*/
	Eigen::VectorXd initFromMeasurement(const Eigen::VectorXd &z);

	/**
	* Normalize measurement if necessary
	* @param z The measurement
	*/
	Eigen::VectorXd normalizeMeasurement(const Eigen::VectorXd &z);

	/**
	* Calc difference between measurement and predicted value
	* @param z The measurement
	* @param predicted Predicted vector
	*/
	Eigen::VectorXd calcMeasurementDiff(const Eigen::VectorXd &z, const Eigen::VectorXd &predicted);

public:

	/**
	* Constructor
	* @param process Kalman filter process
	*/
	ExtKalmanFilterRadar(KalmanFilterProcess &process);

	/**
	* Constructor with matrices
	* @param process Kalman filter process
	* @param R Measurement covariance matrix
	*/
	ExtKalmanFilterRadar(KalmanFilterProcess &process, Eigen::MatrixXd &R);

	/**
	* Destructor
	*/
	virtual ~ExtKalmanFilterRadar();

	/**
	* Initialize Kalman filter measurement with matrices
	* This also mark Kalman filter measurement as initialized
	* @param H Measurement matrix
	* @param R Measurement covariance matrix
	*/
	void Init(Eigen::MatrixXd &H, Eigen::MatrixXd &R);

	/**
	* Initialize Kalman filter measurement with matrices
	* This also mark Kalman filter measurement as initialized
	* @param R Measurement covariance matrix
	*/
	virtual void Init(Eigen::MatrixXd &R);
};


#endif /* KALMAN_FILTER_H_ */
