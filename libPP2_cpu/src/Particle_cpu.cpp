//
///  @file Particle.cpp
///  @brief Particle class that includes all attributes of the particle

#include "include/Particle_cpu.h"

void Particle::updatePosition(float _elapsedtime)
{
  m_position+=m_velocity*_elapsedtime;

  if(m_position[0]>1.0f)
  {
    m_position[0] = 1.0f;
    m_velocity[0]= -m_velocity[0]*0.5f;
  }
  else if(m_position[0]<0.0f)
  {
    m_position[0]= 0.0f;
    m_velocity[0]= -m_velocity[0]*0.5f;
  }
  if(m_position[1]<0.0f)
  {
    m_position[1]=0.0f;
    m_velocity[1]= -m_velocity[1]*0.5f;
  }
  else if (m_position[1]>1.0f)
  {
    m_position[1]=1.0f;
    m_velocity[1]= -m_velocity[1]*0.5f;
  }
}

Vec3 Particle::getPosition() const
{
  return m_position;
}

void Particle::setPosition(Vec3 _pos)
{
  m_position=_pos;
}

void Particle::setVelocity(Vec3 _newvel)
{
  m_velocity = _newvel;
}

Vec3 Particle::getVelocity() const
{
  return m_velocity;
}

void Particle::addVelocity(Vec3 _addedvel)
{
  m_velocity+=_addedvel;
}


void Particle::addPosition(Vec3 _pos)
{
  m_position+=_pos;

  if(m_position[0]>1.0f)
  {
    m_position[0] = 1.0f;
    m_velocity[0] = -m_velocity[0]*0.5f;
  }
  else if(m_position[0]<0.0f)
  {
    m_position[0]= 0.0f;
    m_velocity[0] = -m_velocity[0]*0.5f;
  }
  if(m_position[1]<0.0f)
  {
    m_position[1]=0.0f;
    m_velocity[1] = -m_velocity[1]*0.5f;
  }
  else if (m_position[1]>1.0f)
  {
    m_position[1]=1.0f;
    m_velocity[1] = -m_velocity[1]*0.5f;
  }
}

void Particle::updatePrevPosition()
{
  m_prevPosition=Vec3(m_position[0],m_position[1],m_position[2]);
}

Vec3 Particle::getPrevPosition() const
{
  return m_prevPosition;
}

void Particle::setGridPosition(int _p)
{
  m_gridPosition=_p;
}

int Particle::getGridPosition() const
{
  return m_gridPosition;
}

void Particle::setDrag(bool _drag)
{
  m_dragged=_drag;
}

bool Particle::getDrag() const
{
  return m_dragged;
}

bool Particle::getWall() const
{
  return m_wall;
}

void Particle::setWall(bool _newwall)
{
  m_wall=_newwall;
}

ParticleProperties *Particle::getProperties() const
{
  return m_properties;
}

void Particle::setIsObject()
{
  m_isPartOfObject=true;
}

void Particle::setInit()
{
  m_init=true;
}

bool Particle::isInit()
{
  return m_init;
}

bool Particle::isObject()
{
  return m_isPartOfObject;
}

void Particle::setAlive(bool _i)
{
  m_alive=_i;
}

bool Particle::getAlive()
{
  return m_alive;
}

void Particle::setIndex(int _i)
{
  m_index=_i;
}

int Particle::getIndex()
{
  return m_index;
}

void Particle::updateSpringIndex(int _from, int _to)
{
  for(int i=0; i<(int)m_particleSprings.size(); ++i)
  {
    if(m_particleSprings[i]==_from)
    {
      if(_to>-1) m_particleSprings[i]=_to;
      else m_particleSprings.erase(m_particleSprings.begin()+i);
      break;
    }
  }
}
