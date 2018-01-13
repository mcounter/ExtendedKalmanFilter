#include <stdexcept>
#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

// KalmanFilterProcess
KalmanFilterProcess::KalmanFilterProcess() {}

KalmanFilterProcess::~KalmanFilterProcess() {}

KalmanFilterProcess::KalmanFilterProcess(MatrixXd &P, MatrixXd &F, MatrixXd &Q) : KalmanFilterProcess()
{
	Init(P, F, Q);
}

void KalmanFilterProcess::validateMatrices(MatrixXd &P, MatrixXd &F, MatrixXd &Q)
{
	// Define base sizes
	auto x_size = P.rows();

	// Validations
	if ((x_size <= 0) || (P.cols() != x_size))
	{
		throw std::runtime_error("P matrix must be squared with positive number of columns and rows.");
	}

	if ((F.rows() != x_size) || (F.cols() != x_size))
	{
		throw std::runtime_error("F matrix must be squared with same size as P.");
	}

	if ((Q.rows() != x_size) || (Q.cols() != x_size))
	{
		throw std::runtime_error("Q matrix must be squared with same size as P.");
	}
}

void KalmanFilterProcess::Init(MatrixXd &P, MatrixXd &F, MatrixXd &Q)
{
	initialized_ = false;
	isProcessInitialized = false;

	validateMatrices(P, F, Q);

	P_ = P;
	F_ = F;
	Q_ = Q;

	isProcessInitialized = true;
}

void KalmanFilterProcess::UpdateState(VectorXd &x)
{
	x_ = x;

	initialized_ = true;
}

void KalmanFilterProcess::UpdateState(VectorXd &x, MatrixXd &P)
{
	x_ = x;
	P_ = P;

	initialized_ = true;
}

void KalmanFilterProcess::Predict(const float &dt)
{
	if (!isProcessInitialized)
	{
		throw std::runtime_error("Kalman process is not initialized. Call Init() to initialize it matrices.");
	}

	if (!initialized_)
	{
		//Bypass prediction step before first state initialization
		return;
	}

	F_ = calcTransitionMatrix(dt);
	Q_ = calcTransitionCov(dt);

	VectorXd newX = performTransition(dt);
	MatrixXd FT = F_.transpose();
	MatrixXd newP = F_ * P_ * FT + Q_;

	// In case of success update internal variables
	x_ = newX;
	P_ = newP;
}

MatrixXd KalmanFilterProcess::calcTransitionMatrix(const float &dt)
{
	return F_;
}

MatrixXd KalmanFilterProcess::calcTransitionCov(const float &dt)
{
	return Q_;
}

VectorXd KalmanFilterProcess::performTransition(const float &dt)
{
	VectorXd newX = F_ * x_;
	return newX;
}

// KalmanFilterProcessLinMov
KalmanFilterProcessLinMov::KalmanFilterProcessLinMov() : KalmanFilterProcess() {}

KalmanFilterProcessLinMov::~KalmanFilterProcessLinMov() {}

KalmanFilterProcessLinMov::KalmanFilterProcessLinMov(VectorXd &q, MatrixXd &P) : KalmanFilterProcessLinMov()
{
	Init(q, P);
}

void KalmanFilterProcessLinMov::validateMatrices(VectorXd &q, MatrixXd &P)
{
	// Define base sizes
	auto q_size = q.size();
	auto x_size = P.rows();
	
	// Validations
	if (q_size != 2)
	{
		throw std::runtime_error("q vector must have size 2.");
	}

	if ((x_size != 4) || (P.cols() != x_size))
	{
		throw std::runtime_error("P matrix must be squared with 4 rows and 4 columns.");
	}
}

void KalmanFilterProcessLinMov::Init(MatrixXd &P, MatrixXd &F, MatrixXd &Q)
{
	throw std::runtime_error("Use correct overridden version of this function.");
}

void KalmanFilterProcessLinMov::Init(VectorXd &q, MatrixXd &P)
{
	initialized_ = false;
	isProcessInitialized = false;

	validateMatrices(q, P);

	MatrixXd newF(4, 4);
	newF <<
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;

	MatrixXd newQ(4, 4);
	newQ <<
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0;

	q_ = q;

	KalmanFilterProcess::Init(P, newF, newQ);
}

MatrixXd KalmanFilterProcessLinMov::calcTransitionMatrix(const float &dt)
{
	MatrixXd newF(4, 4);
	newF <<
		1, 0, dt, 0,
		0, 1, 0, dt,
		0, 0, 1, 0,
		0, 0, 0, 1;

	return newF;
}

MatrixXd KalmanFilterProcessLinMov::calcTransitionCov(const float &dt)
{
	double dt2 = dt * dt;
	double dt3 = dt2 * dt;
	double dt4 = dt3 * dt;

	MatrixXd newQ(4, 4);
	newQ <<
		dt4 / 4.0 * q_(0), 0, dt3 / 2.0 * q_(0), 0,
		0, dt4 / 4.0 * q_(1), 0, dt3 / 2.0 * q_(1),
		dt3 / 2.0 * q_(0), 0, dt2 * q_(0), 0,
		0, dt3 / 2.0 * q_(1), 0, dt2 * q_(1);

	return newQ;
}

// KalmanFilterMeasurement
KalmanFilterMeasurement::KalmanFilterMeasurement(KalmanFilterProcess &process)
{
	process_ = &process;
}

KalmanFilterMeasurement::~KalmanFilterMeasurement() {}

KalmanFilterMeasurement::KalmanFilterMeasurement(KalmanFilterProcess &process, MatrixXd &H, MatrixXd &R) : KalmanFilterMeasurement(process)
{
	Init(H, R);
}

void KalmanFilterMeasurement::validateMatrices(MatrixXd &H, MatrixXd &R)
{
	// Define base sizes
	auto z_size = H.rows();

	// Validations
	if (z_size <= 0)
	{
		throw std::runtime_error("H matrix must have positive number of rows.");
	}

	if ((R.rows() != z_size) || (R.cols() != z_size))
	{
		throw std::runtime_error("R matrix must be squared with size equal number of rows in H matrix.");
	}
}

void KalmanFilterMeasurement::Init(MatrixXd &H, MatrixXd &R)
{
	isMeasurementInitialized = false;

	validateMatrices(H, R);

	H_ = H;
	R_ = R;

	isMeasurementInitialized = true;
}

void KalmanFilterMeasurement::Predict(const float &dt)
{
	process_->Predict(dt);
}

void KalmanFilterMeasurement::Update(const VectorXd &z)
{
	if (!isMeasurementInitialized)
	{
		throw std::runtime_error("Kalman measurement is not initialized. Call Init() to initialize it matrices.");
	}

	if (z.size() != H_.rows())
	{
		throw std::runtime_error("Incorrect number of values in measurement vector.");
	}

	const VectorXd normZ = normalizeMeasurement(z);

	if (!process_->initialized_)
	{
		VectorXd newX = initFromMeasurement(normZ);
		process_->UpdateState(newX);
	}
	else
	{
		H_ = calcMeasurementMatrix();
		R_ = calcMeasurementCov();

		VectorXd predictedZ = predictMeasurement();
		VectorXd y = calcMeasurementDiff(normZ, predictedZ);

		
		MatrixXd HT = H_.transpose();
		MatrixXd S = H_ * process_->P_ * HT + R_;
		MatrixXd S1 = S.inverse();
		MatrixXd K = process_->P_ * HT * S1;

		auto x_size = process_->P_.rows();
		MatrixXd I = MatrixXd::Identity(x_size, x_size);

		VectorXd newX = process_->x_ + K * y;
		MatrixXd newP = (I - K * H_) * process_->P_;

		// If not any exceptions happened, update internal variables
		process_->x_ = newX;
		process_->P_ = newP;
	}
}

MatrixXd KalmanFilterMeasurement::calcMeasurementMatrix()
{
	return H_;
}

MatrixXd KalmanFilterMeasurement::calcMeasurementCov()
{
	return R_;
}

VectorXd KalmanFilterMeasurement::predictMeasurement()
{
	VectorXd predictedZ = H_ * process_->x_;
	return predictedZ;
}

VectorXd KalmanFilterMeasurement::initFromMeasurement(const Eigen::VectorXd &z)
{
	MatrixXd HT = H_.transpose();
	MatrixXd HTH = HT * H_;
	MatrixXd HTH1 = HTH.inverse();

	VectorXd newX = HTH1 * HT * z;

	return newX;
}

VectorXd KalmanFilterMeasurement::normalizeMeasurement(const VectorXd &z)
{
	return z;
}

VectorXd KalmanFilterMeasurement::calcMeasurementDiff(const VectorXd &z, const VectorXd &predicted)
{
	return z - predicted;
}

// ExtKalmanFilterLidar
ExtKalmanFilterLidar::ExtKalmanFilterLidar(KalmanFilterProcess &process) : KalmanFilterMeasurement(process) {}

ExtKalmanFilterLidar::~ExtKalmanFilterLidar() {}

ExtKalmanFilterLidar::ExtKalmanFilterLidar(KalmanFilterProcess &process, MatrixXd &R) : ExtKalmanFilterLidar(process)
{
	Init(R);
}

void ExtKalmanFilterLidar::validateMatrices(MatrixXd &R)
{
	// Define base sizes
	auto z_size = R.rows();

	// Validations
	if ((z_size != 2) || (R.cols() != z_size))
	{
		throw std::runtime_error("R matrix must be squared with size 2 rows and 2 columns.");
	}
}

void ExtKalmanFilterLidar::Init(MatrixXd &H, MatrixXd &R)
{
	throw std::runtime_error("Use correct overridden version of this function.");
}

void ExtKalmanFilterLidar::Init(MatrixXd &R)
{
	isMeasurementInitialized = false;

	validateMatrices(R);

	MatrixXd newH = MatrixXd(2, 4);
	newH <<
		1, 0, 0, 0,
		0, 1, 0, 0;

	KalmanFilterMeasurement::Init(newH, R);
}

VectorXd ExtKalmanFilterLidar::initFromMeasurement(const Eigen::VectorXd &z)
{
	VectorXd newX = VectorXd(4);
	newX << z(0), z(1), 0, 0;

	return newX;
}

// ExtKalmanFilterRadar
ExtKalmanFilterRadar::ExtKalmanFilterRadar(KalmanFilterProcess &process) : KalmanFilterMeasurement(process) {}

ExtKalmanFilterRadar::~ExtKalmanFilterRadar() {}

ExtKalmanFilterRadar::ExtKalmanFilterRadar(KalmanFilterProcess &process, MatrixXd &R) : ExtKalmanFilterRadar(process)
{
	Init(R);
}

void ExtKalmanFilterRadar::validateMatrices(MatrixXd &R)
{
	// Define base sizes
	auto z_size = R.rows();

	// Validations
	if ((z_size != 3) || (R.cols() != z_size))
	{
		throw std::runtime_error("R matrix must be squared with size 3 rows and 3 columns.");
	}
}

void ExtKalmanFilterRadar::Init(MatrixXd &H, MatrixXd &R)
{
	throw std::runtime_error("Use correct overridden version of this function.");
}

void ExtKalmanFilterRadar::Init(MatrixXd &R)
{
	isMeasurementInitialized = false;

	validateMatrices(R);

	MatrixXd newH = MatrixXd(3, 4);
	newH <<
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0;

	KalmanFilterMeasurement::Init(newH, R);
}

MatrixXd ExtKalmanFilterRadar::calcMeasurementMatrix()
{
	auto& px = process_->x_(0);
	auto& py = process_->x_(1);
	auto& vx = process_->x_(2);
	auto& vy = process_->x_(3);

	auto pxy11 = px * px + py * py;
	auto pxy12 = sqrt(pxy11);
	auto pxy32 = pxy11 * pxy12;

	if (pxy12 < min_distance_)
	{
		throw std::runtime_error("RADAR measurement too close to point of origin.");
	}

	auto pxxy12 = px / pxy12;
	auto pyxy12 = py / pxy12;
	auto pyvxyyx = py * (vx * py - vy * px);
	auto pxvyxxy = px * (vy * px - vx * py);

	MatrixXd newH = MatrixXd(3, 4);
	newH <<
		pxxy12, pyxy12, 0, 0,
		-py / pxy11, px / pxy11, 0, 0,
		pyvxyyx / pxy32, pxvyxxy / pxy32, pxxy12, pyxy12;

	return newH;
}

VectorXd ExtKalmanFilterRadar::predictMeasurement()
{
	auto& px = process_->x_(0);
	auto& py = process_->x_(1);
	auto& vx = process_->x_(2);
	auto& vy = process_->x_(3);

	auto pxy11 = px * px + py * py;
	auto pxy12 = sqrt(pxy11);

	if (pxy12 < min_distance_)
	{
		throw std::runtime_error("RADAR measurement too close to point of origin.");
	}

	VectorXd predictedZ = VectorXd(3);
	predictedZ << pxy12, atan2(py, px), (px * vx + py * vy) / pxy12;

	return predictedZ;
}

VectorXd ExtKalmanFilterRadar::initFromMeasurement(const Eigen::VectorXd &z)
{
	auto& rho = z(0);
	auto& theta = z(1);
	auto& drho = z(2);

	VectorXd newX = VectorXd(4);
	newX << rho * cos(theta), rho * sin(theta), drho * cos(theta), drho * sin(theta);

	return newX;
}

VectorXd ExtKalmanFilterRadar::normalizeMeasurement(const Eigen::VectorXd &z)
{
	auto fi = z(1);
	if ((fi >= -M_PI) && (fi <= M_PI))
	{
		return z;
	}

	auto pi2 = M_PI + M_PI;

	while (fi < -M_PI)
	{
		fi += pi2;
	}

	while (fi > M_PI)
	{
		fi -= pi2;
	}

	VectorXd newZ = VectorXd(z);
	newZ(1) = fi;

	return newZ;
}

VectorXd ExtKalmanFilterRadar::calcMeasurementDiff(const VectorXd &z, const VectorXd &predicted)
{
	VectorXd y = z - predicted;
	auto& fi = y(1);
	if ((fi >= -M_PI) && (fi <= M_PI))
	{
		return y;
	}

	auto pi2 = M_PI + M_PI;

	while (fi < -M_PI)
	{
		fi += pi2;
	}

	while (fi > M_PI)
	{
		fi -= pi2;
	}

	return y;
}
