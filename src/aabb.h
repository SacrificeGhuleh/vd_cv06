#ifndef AABB_H_
#define AABB_H_

#include "vector3.h"

/*! \class AABB
\brief Obalová struktura.

Obalová struktura v podobě osově zarovnaného kvádru (axis aligned bounding
box), stěny kvádru jsou kolmé na souřadné osy. Rozsah obalové struktury je
určen dvojicí vektorů představující dolní a horní mez.

\author Tomáš Fabián
\version 1.0
\date 2011-2012
*/
class AABB {
public:
  //! Implicitní konstruktor.
  /*!
  Inicializuje meze struktury na
  hodnoty \f$(+\infty,+\infty,+\infty)\f$ až \f$(-\infty,-\infty,-\infty)\f$,
  tj. struktura nic neobsahuje.
  */
  AABB();
  
  //! Specializovaný konstruktor.
  /*!
  Inicializuje meze struktury na hodnoty \a p0 až \a p1.

  \param p0 spodní mez.
  \param p1 horní mez.
  */
  AABB(const Vector3 &p0, const Vector3 &p1);
  
  //! Sloučení dvou obalových struktur.
  /*!
  Nové meze struktury budou obsahovat zadanou \a aabb strukturu.

  \param aabb druhá obalová struktura.
  */
  void Merge(const AABB &aabb);
  
  //! Zahrnutí bodu do obalové struktury.
  /*!
  Obalová struktura bude rozšířena tak, aby obsahovala zadaný bod \a p.

  \param p bod.
  */
  void Merge(const Vector3 &p);
  
  //! Index dominantní osy.
  /*!
  \return Index dominantní osy.
  */
  char dominant_axis() const;
  
  //! Vypočte geometrický střed obalové struktury.
  /*!
  \return Střed obalové struktury.
  */
  Vector3 center() const;
  
  //! Vypočte plochu stěn obalové struktury.
  /*!
  \return Plocha stěn obalové struktury.
  */
  float surface_area() const;
  
  //! Dolní mez obalové struktury.
  /*!
  \return Dolní mez obalové struktury.
  */
  Vector3 lower_bound() const;
  
  //! Horní mez obalové struktury.
  /*!
  \return Horní mez obalové struktury.
  */
  Vector3 upper_bound() const;
  
  Vector3 &operator[](int i) {
    return bounds_[i];
  }

private:
  Vector3 bounds_[2]; /*!< Dolní [0] a horní [1] mez obalové struktury. */
};

#endif
