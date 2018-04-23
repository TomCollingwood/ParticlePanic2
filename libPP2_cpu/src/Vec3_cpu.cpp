///
///  @file Vec3.cpp
///  @brief encapsulates a 3d Point / Vector object. Homogenous is not needed.

#include "include/Vec3_cpu.h"

float Vec3::dot(const Vec3 &_rhs) const
{
  return m_x*_rhs.m_x + m_y*_rhs.m_y + m_z*_rhs.m_z;
}

Vec3 Vec3::cross(Vec3 &_rhs) const
{
  return Vec3(m_y*_rhs.m_z-m_z*_rhs.m_y,
              m_z*_rhs.m_x-m_x*_rhs.m_z,
              m_x*_rhs.m_y-m_y*_rhs.m_x);
}

float Vec3::length() const
{
  return sqrt(m_x*m_x + m_y*m_y + m_z*m_z);
}

float Vec3::lengthSquared() const
{
  return m_x*m_x + m_y*m_y + m_z*m_z;
}

void Vec3::normalize()
{
  float l=length();
  //assert(l != 0.0f);
  if(l!=0.0f)
  {
    m_x/=l;
    m_y/=l;
    m_z/=l;
  }
}

Vec3 Vec3::operator *(GLfloat _rhs) const
{
  return Vec3(m_x * _rhs,
              m_y * _rhs,
              m_z * _rhs
              );
}

void Vec3::operator *=(GLfloat _rhs)
{
  m_x *= _rhs;
  m_y *= _rhs;
  m_z *= _rhs;
}

Vec3 Vec3::operator /(GLfloat _rhs) const
{
  return Vec3(m_x / _rhs,
              m_y / _rhs,
              m_z / _rhs
              );
}

void Vec3::operator /=(GLfloat _rhs)
{
  m_x /= _rhs;
  m_y /= _rhs;
  m_z /= _rhs;
}

Vec3 Vec3::operator +(const Vec3 &_r) const
{
  return Vec3(m_x+_r.m_x,
              m_y+_r.m_y,
              m_z+_r.m_z
              );
}

void Vec3::operator +=(const Vec3 &_r)
{
  m_x += _r.m_x;
  m_y += _r.m_y;
  m_z += _r.m_z;
}

Vec3 Vec3::operator -(const Vec3 &_rhs) const
{
  return Vec3(m_x-_rhs.m_x,
              m_y-_rhs.m_y,
              m_z-_rhs.m_z
              );
}

Vec3 Vec3::operator -()
{
  return Vec3(-m_x,-m_y,-m_z);
}

void Vec3::operator -=(const Vec3 &_r)
{
  m_x -= _r.m_x;
  m_y -= _r.m_y;
  m_z -= _r.m_z;
}

bool Vec3::operator ==(const Vec3 &_rhs) const
{
  if(std::abs(m_x-_rhs.m_x)==0 && std::abs(m_y-_rhs.m_y)==0 && std::abs(m_z-_rhs.m_z)==0) return true;
  else return false;
}

GLfloat & Vec3::operator [](int _i)
{
  assert(_i>=0 && _i<=2);
  return m_openGL[_i];
}

void Vec3::set(GLfloat _x, GLfloat _y, GLfloat _z)
{
  m_x=_x;
  m_y=_y;
  m_z=_z;
}

void Vec3::vertexGL() const
{
  glVertex3f(m_x,m_y,m_z);
}
