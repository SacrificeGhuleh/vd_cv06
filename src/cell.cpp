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
  // TODO compute the gradient of the scalar field here (use finite central differences or Sobel–Feldman operator)
  
  return Vector3();
}

float Cell::Integrate(Ray &ray, const float t0, const float t1) const {
  // TODO approximate computation of an integral of scalar values along the given segment of the ray
  
  return 0.0f;
}

float Cell::FindIsoSurface(Ray &ray, const float t0, const float t1, const float iso_value) const {
  // TODO find the parametric distance of the iso surface of the certain iso value along the given segment of the ray
  
  return -1.0f;
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
