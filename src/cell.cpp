#include "stdafx.h"

#include "cell.h"

Cell::Cell(const DATA_TYPE *rhos, const Vector3 &A, const Vector3 &B) {
  memcpy(rhos_, rhos, sizeof(rhos_));
  
  bounds_ = AABB(A, B);
}

Cell::~Cell() {

}

float Cell::Gamma(const Vector3 &uvw) const {
  //return ( rho_A() + rho_B() + rho_C() + rho_D() + rho_E() + rho_F() + rho_G() + rho_H() ) / 8.0f;
  
  const float &u_t = uvw.x;
  const float &v_t = uvw.y;
  const float &w_t = uvw.z;
  
  // trilinear interpolation
  const COMP_TYPE alpha_AB = rho_A() * (1 - u_t) + rho_B() * u_t;
  const COMP_TYPE alpha_DC = rho_D() * (1 - u_t) + rho_C() * u_t;
  const COMP_TYPE alpha_EF = rho_E() * (1 - u_t) + rho_F() * u_t;
  const COMP_TYPE alpha_HG = rho_H() * (1 - u_t) + rho_G() * u_t;
  
  const COMP_TYPE beta_0 = alpha_AB * (1 - v_t) + alpha_DC * v_t;
  const COMP_TYPE beta_1 = alpha_EF * (1 - v_t) + alpha_HG * v_t;
  
  return static_cast< DATA_TYPE >( beta_0 * (1 - w_t) + beta_1 * w_t );
  
  // TODO try to add tricubic interpolation
}

Vector3 Cell::GradGamma(const Vector3 &uvw) const {
//  const int sobelX[3][3] = {
//      {1, 0, -1},
//      {2, 0, -2},
//      {1, 0, -1}
//  };
//
//  const int sobelY[3][3] = {
//      {1,  2,  1},
//      {0,  0,  0},
//      {-1, -2, -1}
//  };
//
//  const std::pair<int, int> positions[3][3] = {
//      {{-1, -1}, {0, -1}, {1, -1}},
//      {{-1, -0}, {0, -0}, {1, 0}},
//      {{-1, -1}, {0, -1}, {1, 1}},
//  };
//
//  Vector3 Gx(0, 0, 0);
//  Vector3 Gy(0, 0, 0);
//
//  for (int row = 0; row < 3; row++) {
//    for (int col = 0; col < 3; col++) {
//      const std::pair<int, int> pos = positions[col][row];
//      const int xSobelVal = sobelX[col][row];
//      const int ySobelVal = sobelY[col][row];
//
//
//    }
//  }
  
  const float derivativeDistance = 0.001f;
  
  const float x1 = Gamma(uvw - Vector3(derivativeDistance, .0f, .0f));
  const float x2 = Gamma(uvw - Vector3(-derivativeDistance, .0f, .0f));
  const float y1 = Gamma(uvw - Vector3(.0f, derivativeDistance, .0f));
  const float y2 = Gamma(uvw - Vector3(.0f, -derivativeDistance, .0f));
  const float z1 = Gamma(uvw - Vector3(.0f, .0f, derivativeDistance));
  const float z2 = Gamma(uvw - Vector3(.0f, .0f, -derivativeDistance));
  
  const float dx = x1 - x2;
  const float dy = y1 - y2;
  const float dz = z1 - z2;
  
  Vector3 normal = Vector3(dx, dy, dz) * (1.0f / (2.0f * derivativeDistance));
  normal.Normalize();
  return normal;
}

float Cell::Integrate(Ray &ray, const float t0, const float t1) const {
  // TODO approximate computation of an integral of scalar values along the given segment of the ray
  float integrationResult = 0.f;
  
  // Number of samples for integration computation
  const int integResolution = 10;
  // Delta of sample
  const float dt = (1.0f / integResolution) * (t1 - t0);
  for (int i = 0; i < integResolution; i++) {
    const float t = t0 + (i * dt);
    const float f = Gamma(u(ray.eval(t)));
    integrationResult += f * dt;
  }
  
  return integrationResult;
}

float Cell::FindIsoSurface(Ray &ray, const float t0, const float t1, const float iso_value) const {
  // TODO find the parametric distance of the iso surface of the certain iso value along the given segment of the ray
  const float dt = 0.01f;
  float t = t0;
  while (t < t1) {
    const float f = Gamma(u(ray.eval(t)));
    if (f > iso_value) {
      return t;
    }
    t += dt;
  }
  return -1.f;
}

Vector3 Cell::u(const Vector3 &p) const {
  Vector3 uvw = (p - A()) / (G() - A()); // gives the reference coordinates of the world space point p inside this cell (uvw is in the range <0,1>^3)
  
  return uvw;
}

Vector3 Cell::A() const {
  return bounds_.lower_bound();
}

Vector3 Cell::G() const {
  return bounds_.upper_bound();
}

float Cell::rho_A() const {
  return rhos_[0];
}

float Cell::rho_B() const {
  return rhos_[1];
}

float Cell::rho_C() const {
  return rhos_[2];
}

float Cell::rho_D() const {
  return rhos_[3];
}

float Cell::rho_E() const {
  return rhos_[4];
}

float Cell::rho_F() const {
  return rhos_[5];
}

float Cell::rho_G() const {
  return rhos_[6];
}

float Cell::rho_H() const {
  return rhos_[7];
}
