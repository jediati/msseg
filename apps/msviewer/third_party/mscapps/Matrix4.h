#ifndef MATRIX4
#define MATRIX4

#include <math.h>
#include <memory.h>
#include <stdio.h>
#define PI 3.14159


struct Matrix4 {
	float vals[16];
};

struct Vector4 {
	float vals[4];

	Vector4() { for (int i = 0; i < 4; i++) vals[i] = 0; }
	Vector4(float x, float y, float z, float w) {
		vals[0] = x;
		vals[1] = y;
		vals[2] = z;
		vals[3] = w;
	}

	Vector4 operator=(const Vector4& other){
		this->vals[0] = other.vals[0];
		this->vals[1] = other.vals[1];
		this->vals[2] = other.vals[2];
		this->vals[3] = other.vals[3];
		return *this;
	}
	void operator+=(const Vector4& other) {
		this->vals[0] += other.vals[0];
		this->vals[1] += other.vals[1];
		this->vals[2] += other.vals[2];
		this->vals[3] += other.vals[3];
	}
	void operator-=(const Vector4& other) {
		this->vals[0] -= other.vals[0];
		this->vals[1] -= other.vals[1];
		this->vals[2] -= other.vals[2];
		this->vals[3] -= other.vals[3];
	}

	void operator-=(const float other) {
		this->vals[0] -= other;
		this->vals[1] -= other;
		this->vals[2] -= other;
		this->vals[3] -= other;
	}
	void operator*=(const float other) {
		this->vals[0] *= other;
		this->vals[1] *= other;
		this->vals[2] *= other;
		this->vals[3] *= other;
	}

	float MagSq() const {
		return (vals[0] * vals[0]) + (vals[1] * vals[1]) + (vals[2] * vals[2]) + (vals[3] * vals[3]);
	}

	float Mag() const {
		return sqrt(MagSq());
	}
	Vector4 operator+(const Vector4  &a) const {
		Vector4 res(*this);
		res += a;
		return res;
	}
	Vector4 operator-(const Vector4  &a) const {
		Vector4 res(*this);
		res -= a;
		return res;
	}
	void print() const {
		printf("(%f, %f, %f, %f)", vals[0], vals[1], vals[2], vals[3]);
	}
	Vector4 operator+(const float  val) const {
		Vector4 res(*this);
		res.vals[0] += val; res.vals[1] += val;
		res.vals[2] += val; res.vals[3] += val;
		return res;
	}
	Vector4 operator-(const float  val) const {
		Vector4 res(*this);
		res.vals[0] -= val; res.vals[1] -= val;
		res.vals[2] -= val; res.vals[3] -= val;
		return res;
	}

	Vector4 operator*(const float  val) const {
		Vector4 res(this->vals[0], this->vals[1], this->vals[2], this->vals[3]);// = (*this);
		res.vals[0] *= val; res.vals[1] *= val;
		res.vals[2] *= val; res.vals[3] *= val;
		return res;
	}
	bool operator==(const Vector4& other) const {
		return memcmp(this, &other, sizeof(Vector4)) == 0;
	}

	float& operator[](int i) { return vals[i]; }
	const float& operator[](int i) const { return vals[i]; }


	static Vector4 piecewiseMax(const Vector4& a, const Vector4& b) {
		Vector4 c(a);
		for (int i = 0; i < 4; i++) if (c[i] < b[i]) c[i] = b[i];
		return c;
	}
	static Vector4 piecewiseMin(const Vector4& a, const Vector4& b) {
		Vector4 c(a);
		for (int i = 0; i < 4; i++) if (c[i] > b[i]) c[i] = b[i];
		return c;
	}
};

float ToRadians(float degrees);
float ToDegrees(float radians);
float Dot3(Vector4 A, Vector4 B);
float InteriorAngle(Vector4 A, Vector4 B);
int Abs3(Vector4* v);
Vector4 Scale(Vector4 v, float s);
int Normalize3(Vector4* v);
void Add3(Vector4& a, Vector4& b, Vector4& res);
void Normalize4(Vector4* v);
Vector4 Cross(Vector4 a, Vector4 b);
Vector4 GMVMf(Matrix4 mat, Vector4 vect);
Matrix4 GMMMf(Matrix4 A, Matrix4 B);
Matrix4 Translate3f(float x, float y, float z);
Matrix4 GMSMf(Matrix4 A, float b);
Matrix4 GMMAf(Matrix4 A, Matrix4 B);
Matrix4 MIdentity();
Matrix4 Transpose(Matrix4 m);
// rotate around UNIT vect a around origin, alpha degrees
Matrix4 RoatateAVO(float alpha, Vector4 a);

struct Quaternion {
	float x, y, z, w;

	static Quaternion times(Quaternion a, Quaternion b) {
		Quaternion res;
		res.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
		res.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
		res.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
		res.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
		return res;
	}

	static Quaternion rotation(Vector4 v, float angle) {
		Quaternion res;
		Normalize3(&v);
		res.w = cosf(angle / 2.0);
		float tmp = sinf(angle / 2.0);
		res.x = v.vals[0] * tmp;
		res.y = v.vals[1] * tmp;
		res.z = v.vals[2] * tmp;
		return res;
	}
	static void setRotQuat(Matrix4 &m, Quaternion &q) {
			{
				float x2 = 2.0f * q.x, y2 = 2.0f * q.y, z2 = 2.0f * q.z;

				float xy = x2 * q.y, xz = x2 * q.z;
				float yy = y2 * q.y, yw = y2 * q.w;
				float zw = z2 * q.w, zz = z2 * q.z;

				m.vals[0] = 1.0f - (yy + zz);
				m.vals[1] = (xy - zw);
				m.vals[2] = (xz + yw);
				m.vals[3] = 0.0f;

				float xx = x2 * q.x, xw = x2 * q.w, yz = y2 * q.z;

				m.vals[4] = (xy + zw);
				m.vals[5] = 1.0f - (xx + zz);
				m.vals[6] = (yz - xw);
				m.vals[7] = 0.0f;

				m.vals[8] = (xz - yw);
				m.vals[9] = (yz + xw);
				m.vals[10] = 1.0f - (xx + yy);
				m.vals[11] = 0.0f;

				m.vals[12] = 0.0f;
				m.vals[13] = 0.0f;
				m.vals[14] = 0.0f;
				m.vals[15] = 1.0f;
			}

	}
	static void resetQuat(Quaternion &q) {
		q.x = 0;
		q.y = 0;
		q.z = 0;
		q.w = 1;
	}
};
struct Matrix4D {
	double vals[16];
};

struct Vector4D {
	double vals[4];

	Vector4D(double x, double y, double z, double w) {
		vals[0] = x; vals[1] = y; vals[2] = z; vals[3] = w;
	}
	Vector4D() {}

	void operator=(const Vector4D& other){
		this->vals[0] = other.vals[0];
		this->vals[1] = other.vals[1];
		this->vals[2] = other.vals[2];
		this->vals[3] = other.vals[3];
	}
	void operator+=(const Vector4D& other) {
		this->vals[0] += other.vals[0];
		this->vals[1] += other.vals[1];
		this->vals[2] += other.vals[2];
		this->vals[3] += other.vals[3];
	}
	void operator-=(const Vector4D& other) {
		this->vals[0] -= other.vals[0];
		this->vals[1] -= other.vals[1];
		this->vals[2] -= other.vals[2];
		this->vals[3] -= other.vals[3];
	}

	void operator-=(const double other) {
		this->vals[0] -= other;
		this->vals[1] -= other;
		this->vals[2] -= other;
		this->vals[3] -= other;
	}
	void operator*=(const double other) {
		this->vals[0] *= other;
		this->vals[1] *= other;
		this->vals[2] *= other;
		this->vals[3] *= other;
	}

	double MagSq() const {
		return (vals[0] * vals[0]) + (vals[1] * vals[1]) + (vals[2] * vals[2]) + (vals[3] * vals[3]);
	}

	double Mag() const {
		return sqrt(MagSq());
	}
	Vector4D operator+(const Vector4D  &a) const {
		Vector4D res(*this);
		res += a;
		return res;
	}
	Vector4D operator-(const Vector4D  &a) const {
		Vector4D res(*this);
		res -= a;
		return res;
	}

	Vector4D operator+(const double  val) const {
		Vector4D res(*this);
		res.vals[0] += val; res.vals[1] += val;
		res.vals[2] += val; res.vals[3] += val;
		return res;
	}
	Vector4D operator-(const double  val) const {
		Vector4D res(*this);
		res.vals[0] -= val; res.vals[1] -= val;
		res.vals[2] -= val; res.vals[3] -= val;
		return res;
	}

	Vector4D operator*(const double  val) const {
		Vector4D res(this->vals[0], this->vals[1], this->vals[2], this->vals[3]);// = (*this);
		res.vals[0] *= val; res.vals[1] *= val;
		res.vals[2] *= val; res.vals[3] *= val;
		return res;
	}
	bool operator==(const Vector4D& other) const {
		return memcmp(this, &other, sizeof(Vector4D)) == 0;
	}
};

double ToRadiansD(double degrees);
double ToDegreesD(double radians);
double Dot3D(Vector4D A, Vector4D B);
double InteriorAngleD(Vector4D A, Vector4D B);
int Abs3D(Vector4D* v);
Vector4D ScaleD(Vector4D v, double s);
int Normalize3D(Vector4D* v);
void Add3D(Vector4D& a, Vector4D& b, Vector4D& res);
void Normalize4D(Vector4D* v);
Vector4D CrossD(Vector4D a, Vector4D b);
Vector4D GMVMD(Matrix4D mat, Vector4D vect);
Matrix4D GMMMD(Matrix4D A, Matrix4D B);
Matrix4D Translate3D(double x, double y, double z);
Matrix4D GMSMD(Matrix4D A, double b);
Matrix4D GMMAD(Matrix4D A, Matrix4D B);
Matrix4D MIdentityD();
Matrix4D TransposeD(Matrix4D m);
// rotate around UNIT vect a around origin, alpha degrees
Matrix4D RoatateAVOD(double alpha, Vector4D a);
void PrintMatrix(Matrix4 m);

#endif