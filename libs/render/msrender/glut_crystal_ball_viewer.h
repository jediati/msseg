#ifndef GLUT_CRYSTAL_BALL_VIEWER
#define GLUT_CRYSTAL_BALL_VIEWER

#define GLUT_DISABLE_ATEXIT_HACK

#define MAXVOLSIZE 512ll*512ll*512ll*3ll

#include <math.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#ifdef WIN32
#include <windows.h>
#endif

#include "vsvr2.h"
#include "Matrix4.h"
#include "freeglut.h"
#include "glsl.h"

#define MAX_LINE_LENGTH 5000
#define TFSIZE 512

class RenderableGL {
public:
	virtual void Render() {}
	virtual void BBox(Vector4& vmin, Vector4& vmax) {}
};

class SceneGL : public RenderableGL {
protected:
	std::vector<RenderableGL*> mRenderObjects;

public:

	virtual void BBox(Vector4& vmin, Vector4& vmax) {
		if (mRenderObjects.size() == 0) {
			vmin = Vector4(0, 0, 0, 0);
			vmax = Vector4(1, 1, 1, 0);
		}
		Vector4 tmpmin, tmpmax;
		mRenderObjects[0]->BBox(vmin, vmax);
		for (int i = 1; i < mRenderObjects.size(); i++) {
			mRenderObjects[i]->BBox(tmpmin, tmpmax);
			vmin = Vector4::piecewiseMax(vmin, tmpmin);
			vmax = Vector4::piecewiseMax(vmax, tmpmax);
		}
	}

	virtual void Render() {
		for (auto r : mRenderObjects) r->Render();
	}

	virtual void AddRenderable(RenderableGL* r) {
		mRenderObjects.push_back(r);
	}
};


class TFManipulator {
protected:
	float* tfunc;
	float m_minval;
	float m_maxval;

public:
	TFManipulator(float minval, float maxval) : m_minval(minval), m_maxval(maxval) {
		tfunc = new float[TFSIZE * 4];
		for (int i = 0; i < TFSIZE * 4; i++) tfunc[i] = 0;
	}

	void ZeroAll() {
		for (int i = 0; i < TFSIZE * 4; i++) tfunc[i] = 0;
	}

	void SetAll(float r, float g, float b, float a) {
		for (int i = 0; i < TFSIZE; i++) {
			tfunc[i * 4] = r;
			tfunc[i * 4 + 1] = g;
			tfunc[i * 4 + 2] = b;
			tfunc[i * 4 + 3] = a;
		}
	}
	void SetID(int i, float r, float g, float b, float a) {		
			tfunc[i * 4] = r;
			tfunc[i * 4 + 1] = g;
			tfunc[i * 4 + 2] = b;
			tfunc[i * 4 + 3] = a;
	}
	void AddID(int i, float r, float g, float b, float a) {
		tfunc[i * 4] += r;
		tfunc[i * 4 + 1] += g;
		tfunc[i * 4 + 2] += b;
		tfunc[i * 4 + 3] += a;
	}
	void SetIso(float r, float g, float b, float a, 
		float val, int width, bool dark = true) {
		int id = (int)(((float)TFSIZE) * (val - m_minval) / (m_maxval - m_minval));
		int sid = id - width; 
		if (sid < 0) sid = 0;
		int eid = id + width;
		if (eid >= TFSIZE) eid = TFSIZE - 1;
		for (int i = sid; i < id; i++) {
			if (dark) SetID(i, 0, 0, 0, a);
			else SetID(i, 1, 1, 1, a);
		}
		for (int i = id ; i < eid; i++) {
			SetID(i, r, g, b, a);
		}

	}

	void SetSoftIso(float r, float g, float b, float a,
		float val, int lwidth, int uwidth, bool dark = true) {
		int id = (int)(((float)TFSIZE) * (val - m_minval) / (m_maxval - m_minval));
		int sid = id - lwidth;
		if (sid < 0) sid = 0;
		int eid = id + uwidth;
		if (eid >= TFSIZE) eid = TFSIZE - 1;
		for (int i = sid + 1; i < id; i++) {
			float alpha = 0.3 * ((float)(i - sid) / (float)(lwidth));
			if (dark) SetID(i, 0, 0, 0, alpha);
			else SetID(i, 1, 1, 1, alpha );
		}
		for (int i = id; i < eid; i++) {
			float alpha = a * ((float)(eid - i) / (float)(uwidth));
			SetID(i, r, g, b, alpha);
		}
	}
	float* raw_tf() const {
		return tfunc;
	}
};

class VolumeRenderableGL : public RenderableGL {
protected:

	// x y z dim of dataset
	unsigned int X;
	unsigned int Y;
	unsigned int Z;

	// x y z of my volume render volume
	unsigned int sX;
	unsigned int sY;
	unsigned int sZ;

	// values storing the minimum and maximum values in the dataset
	//float maxvalue;
	//float minvalue;

	float tmpmin;
	float tmpmax;
	// the global array storing the scalar values (as floats) of the 
	// dataset
	float* msnv_data;
	float* tfunc;
	VSVR2 *vsvr;
	bool m_rerender;
	int m_numslices;
	const char* m_tf_name;
	bool b_render;
	float _tf_min;
	float _tf_max;

public:

	void DisableRender() {
		b_render = false;
	}
	void EnableRender() {
		b_render = true;
	}
	void ToggleRender() {
		b_render = !b_render;
	}
	bool isOn() {
		return b_render;
	}

	void UpdateTF(float* tf) {
		for (int i = 0; i < TFSIZE * 4; i++)  tfunc[i] = tf[i];
		vsvr->tf_glload();
	}

	void SetRange(float minimum, float maximum) {
		_tf_min = minimum;
		_tf_max = maximum;
		vsvr->SetMinMax(minimum, maximum);
	}
	void ReadTF() {
		//printf("asdf1\n");
		//FILE *f = fopen("transferf/transfer.tf","r");
		FILE *f = fopen(m_tf_name, "r");
		if (!f) { printf("no transfer function %s\n", m_tf_name); return; }
		//printf("i get here... tf found\n");

		//int X = 256; int Y = 256; int Z = 128;
		unsigned int ar; unsigned int ag; unsigned int ab; unsigned int aa;
		int tmpasdf;
		fscanf(f, "%d\n", &tmpasdf);
		for (int i = 0; i < TFSIZE; i++) {
			fscanf(f, "%d, %d, %d, %d\n", &ar, &ag, &ab, &aa);
			//printf("%d, %d, %d, %d\n", ar, ag, ab, aa);

			tfunc[i * 4] = (float)ar / ((float)TFSIZE - 1);
			tfunc[i * 4 + 1] = (float)ag / ((float)TFSIZE - 1);
			tfunc[i * 4 + 2] = (float)ab / ((float)TFSIZE - 1);
			tfunc[i * 4 + 3] = (float)aa / ((float)TFSIZE - 1);
			tfunc[i * 4 + 3] /= 2.0;
			//.printf("%f, %f, %f, %f\n", tfunc[4*i],tfunc[4*i +1],tfunc[4*i + 2], tfunc[4*i + 3]);
			//tfunc[i + 0] = 1;
			//tfunc[i + 1*256] = 0;
			//tfunc[i + 2*256] = 0; 
			//tfunc[i + 3*256] = .2;
		}
		fclose(f);
		//vsvr->tf_glload();
	}

	void ReloadTF() {
		ReadTF();

		vsvr->tf_glload();
	}

	virtual void BBox(Vector4& vmin, Vector4& vmax) {
		vmin = Vector4(0, 0, 0, 0);
		vmax = Vector4(X, Y, Z, 0);
	}

	void SwapFunction(float* data) {
		long long xl = X;
		long long yl = Y;
		long long zl = Z;
		long long osize = xl * yl * zl;
		//minvalue = data[0];
		//maxvalue = data[0];
		//for (long long i = 1; i < osize; i++) {
		//	if (data[i] < minvalue) minvalue = data[i];
		//	if (data[i] > maxvalue) maxvalue = data[i];
		//}
		//float oneoverdiff = 1.0 / (maxvalue - minvalue);
		int stridex, stridey, stridez;
		stridex = stridey = stridez = 1;

		while ((xl / stridex) * (yl / stridey) * (zl / stridez) > MAXVOLSIZE) {
			if ((xl / stridex) >= (yl / stridey) && (xl / stridex) >= (zl / stridez)) {
				stridex *= 2;
				continue;
			}
			else if ((yl / stridey) >= (xl / stridex) && (yl / stridey) >= (zl / stridez)) {
				stridey *= 2;
				continue;
			}
			else {
				stridez *= 2;
				continue;
			}
		}
		printf("using volume rendering strides : x=%d, y=%d, z=%d\n", stridex, stridey, stridez);
		sX = X / stridex;
		sY = Y / stridey;
		sZ = Z / stridez;

		long long newsize = sX * sY * sZ;
		//msnv_data = new float[newsize];
		for (long long tk = 0; tk < sZ; tk++) {
			for (long long tj = 0; tj < sY; tj++) {
				for (long long ti = 0; ti < sX; ti++) {
					msnv_data[ti + tj * sX + tk * sX * sY] = data[(ti * stridex) + (tj * stridey) * X + (tk * stridez) * X * Y];
						//( - minvalue) * oneoverdiff;
				}
			}
		}

		printf("setting new data\n");
		vsvr->tex_change(msnv_data);

	}
	void Initialize(int x, int y, int z, float* data, const char* tfuncname) {

		X = x; Y = y; Z = z; m_tf_name = tfuncname;
		m_numslices = 150;
		m_rerender = true;
		b_render = false;
		// create volume renderable version of normalized data;
		long long xl = x;
		long long yl = y; 
		long long zl = z;

		long long osize = xl*yl*zl;

		/*minvalue = data[0];
		maxvalue = data[0];
		for (long long i = 1; i < osize; i++) {
			if (data[i] < minvalue) minvalue = data[i];
			if (data[i] > maxvalue) maxvalue = data[i];
		}
		float oneoverdiff = 1.0 / ( maxvalue - minvalue);*/

		int stridex, stridey, stridez;
		stridex = stridey = stridez = 1;

		while ((xl / stridex) * (yl / stridey) * (zl / stridez) > MAXVOLSIZE) {
			if ((xl / stridex) >= (yl / stridey) && (xl / stridex) >= (zl / stridez)) {
				stridex *= 2;
				continue;
			}
			else if ((yl / stridey) >= (xl / stridex) && (yl / stridey) >= (zl / stridez)) {
				stridey *= 2;
				continue;
			}
			else {
				stridez *= 2;
				continue;
			}
		}
		printf("using volume rendering strides : x=%d, y=%d, z=%d\n", stridex, stridey, stridez);
		sX = X / stridex;
		sY = Y / stridey;
		sZ = Z / stridez;

		long long newsize = sX * sY * sZ;
		msnv_data = new float[newsize];
		for (long long tk = 0; tk < sZ; tk++) {
			for (long long tj = 0; tj < sY; tj++) {
				for (long long ti = 0; ti < sX; ti++){
					msnv_data[ti + tj * sX + tk * sX * sY] = data[(ti * stridex) + (tj * stridey) * X + (tk * stridez) * X * Y];
						//( - minvalue) * oneoverdiff;
				}
			}
		}
		//for (int i = 0; i < newsize; i++) if (msnv_data[i] > 0.1) printf("%f\n", msnv_data[i]);

		tfunc = new float[TFSIZE * 4];
		printf("about to read tf\n");
		ReadTF();
		printf("read tf, making vsvr\n");
		// Very simple volume rendering
		vsvr = new VSVR2();
		vsvr->tex_set_resolution(sX, sY, sZ);
		vsvr->tex_set_extern(msnv_data);
		vsvr->tf_set_size(TFSIZE);
		vsvr->tf_set_extern(tfunc);
	}

	virtual void SetNumberOfSlices(int num) {
		m_numslices = num;
		//m_rerender = true;
	}
	float min_val() const {
		return _tf_min;
	}
	float max_val() const {
		return _tf_max;
	}

	float num_tf_bins() const {
		return TFSIZE;
	}

	virtual void Render() {
		//return;
		if (!b_render) return;
		//printf("asdf %d\n", m_numslices);
		glPushMatrix();

		glScalef(X / (float)sX, Y / (float)sY, Z / (float)sZ);
		//glLoadIdentity();
		if (m_rerender)
		{
			//ReadTF();
			printf("rerendering from scratch...\n");
			vsvr->gl_render(m_numslices);                // Redraws from scratch
			m_rerender = false;

		}
		else
		{
			//printf("asdf2\n");
			vsvr->gl_redisplay(m_numslices);             // Redraws from display list
		}
		glPopMatrix();
		// 
		//bForceRender is a flag to update the volume rendering due to major changes.
		// 
	}
};


class VolumeSceneGL : public SceneGL {
protected:
	VolumeRenderableGL* mVol;
public:
	virtual void BBox(Vector4& vmin, Vector4& vmax) {
		mVol->BBox(vmin, vmax);
	}

	virtual void Render() {
		SceneGL::Render();
		mVol->Render();
	}

	virtual void AddRenderable(RenderableGL* r) {
		this->mRenderObjects.push_back(r);
	}
};


namespace CRYSTAL_BALL_VIEWER {
	void Initialize_GLUT(int argc, char **argv);
	void StartGlutMainLoop();
	void SetGraphicKeysFunction(bool(*callback)(unsigned char, int, int));
	void SetTimerFunction(unsigned int time, void(*callback)(int), int value);
	void SetMouseMovedFunction(bool(*callback)(int, int));
	void SetMouseFunction(bool(*callback)(int, int, int, int));
	void SetScene(RenderableGL* scene);
	void SetPassiveMotionFunc(bool(*func)(int , int));
	void Redisplay();
}

#endif