///
///  @file World.cpp
///  @brief contains all particles and methods to update them and export

#include "include/World_cpu.h"

WorldCPU::WorldCPU()
{
    initData();
}

WorldCPU::WorldCPU(int _num_points, float _iRadius, float _timestep, int _gridRes)
{
    m_num_points=_num_points;
    m_interactionradius=_iRadius;
    m_timestep=_timestep;
    m_gridResolution=_gridRes;
    initData();
}

WorldCPU::~WorldCPU()
{}

void WorldCPU::initData()
{
    m_particles.clear();
    m_gravity = true;

    // DEFAULT PARTICLE PROPERTIES
    m_particleTypes.push_back(ParticleProperties()); //water

    Particle defaultparticle(Vec3(0.0f,0.0f,0.0f),&m_particleTypes[0]);
    m_particles.resize(m_num_points,defaultparticle);

    srand(42);
    if(m_num_points<=100)
    {
        //------------------DAMBREAKER 100----------------------
        for(int x = 0; x<5; ++x)
        {
            for(int y =0; y<20; ++y)
            {
                if(x+y*5 >= m_num_points) break;
                float xr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                float yr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                xr = 0.5f-xr;
                yr = 0.5f-yr;
                m_particles[x+y*5].setPosition(Vec3(float(x)*(1.0f/20.0f)+xr*0.01f,
                                                    float(y)*(1.0f/20.0f)+yr*0.01f,
                                                    0.0f));
            }
        }
    }
    else if(m_num_points<=10000)
    {
        //------------------DAMBREAKER 10,000----------------------
        for(int x = 0; x<50; ++x)
        {
            for(int y =0; y<200; ++y)
            {
                if(x+y*50 >= m_num_points) break;
                float xr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                float yr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                xr = 0.5f-xr;
                yr = 0.5f-yr;
                m_particles[x+y*50].setPosition(Vec3(float(x)*(1.0f/200.0f)+xr*0.001f,
                                                     float(y)*(1.0f/200.0f)+yr*0.001f,
                                                     0.0f));
            }
        }
    }
    else if(m_num_points<=1000000)
    {
        //------------------DAMBREAKER 1,000,000----------------------
        for(int x = 0; x<500; ++x)
        {
            for(int y =0; y<2000; ++y)
            {
                if(x+y*500 >= m_num_points) break;
                float xr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                float yr = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                xr = 0.5f-xr;
                yr = 0.5f-yr;
                m_particles[x+y*500].setPosition(Vec3(float(x)*(1.0f/2000.0f)+xr*0.0001f,
                                                      float(y)*(1.0f/2000.0f)+yr*0.0001f,
                                                      0.0f));
            }
        }
    }
}

void WorldCPU::dumpToGeo(const uint _cnt)
{
    char fname[150];

    std::sprintf(fname,"geo/SPH_CPU.%03d.geo",_cnt);
    // we will use a stringstream as it may be more efficient
    std::stringstream ss;
    std::ofstream file;
    file.open(fname);
    if (!file.is_open())
    {
        std::cerr << "failed to Open file "<<fname<<'\n';
        exit(EXIT_FAILURE);
    }
    // write header see here http://www.sidefx.com/docs/houdini15.0/io/formats/geo
    ss << "PGEOMETRY V5\n";
    ss << "NPoints " << m_num_points << " NPrims 1\n";
    ss << "NPointGroups 0 NPrimGroups 1\n";
    // this is hard coded but could be flexible we have 1 attrib which is Colour
    ss << "NPointAttrib 1  NVertexAttrib 0 NPrimAttrib 2 NAttrib 0\n";
    // now write out our point attrib this case Cd for diffuse colour
    ss <<"PointAttrib \n";
    // default the colour to white
    ss <<"Cd 3 float 1 1 1\n";
    // now we write out the particle data in the format
    // x y z 1 (attrib so in this case colour)
    for(unsigned int i=0; i<m_num_points; ++i)
    {
        ss<<m_particles[i].getPosition()[0]<<" "<<m_particles[i].getPosition()[1]<<" "<<0 << " 1 ";
        ss<<"("<<1<<" "<<1<<" "<<1<<")\n";
    }

    // now write out the index values
    ss<<"PrimitiveAttrib\n";
    ss<<"generator 1 index 1 location1\n";
    ss<<"dopobject 1 index 1 /obj/AutoDopNetwork:1\n";
    ss<<"Part "<<m_num_points<<" ";
    for(size_t i=0; i<m_num_points; ++i)
    {
        ss<<i<<" ";
    }
    ss<<" [0	0]\n";
    ss<<"box_object1 unordered\n";
    ss<<"1 1\n";
    ss<<"beginExtra\n";
    ss<<"endExtra\n";
    // dump string stream to disk;
    file<<ss.rdbuf();
    file.close();
}

//-----------------------------------------------UPDATE-----------------------------------------

void WorldCPU::simulate(int _substeps)
{
    for(int i =0; i<_substeps; ++i)
    {
        // ------------------------------GRAVITY --------------------------------------------
        if(m_gravity)
        {
            Vec3 gravityvel = Vec3(0.0f,-0.008,0.0f);

            for(int i=0; i<m_num_points; ++i)
            {
                m_particles[i].addVelocity(gravityvel);
            }
        }

        // ------------------------------VISCOSITY--------------------------------------------
        int choo = 0;

        for(auto k = 0; k<(int)m_grid.size(); ++k)
        {
            int ploo = 0;
            for(auto& i : m_grid[k])
            {
                if(!(i->getWall()))
                {
                    std::vector<Particle *> surroundingParticles = getSurroundingParticles(choo,1,false);
                    int cloo = 0;
                    for(auto& j : surroundingParticles)
                    {
                        if(cloo>ploo && !(j->getWall()))
                        {
                            Vec3 rij=(j->getPosition()-i->getPosition());
                            float q = rij.length()/m_interactionradius;
                            if(q<1 && q!=0)
                            {
                                rij.normalize();
                                float u = (i->getVelocity()-j->getVelocity()).dot(rij);
                                if(u>0)
                                {
                                    ParticleProperties *thisproperties = i->getProperties();
                                    float sig = thisproperties->getSigma();
                                    float bet = thisproperties->getBeta();
                                    Vec3 impulse = rij*((1-q)*(sig*u + bet*u*u))*m_timestep;
                                    i->addVelocity(-impulse/2.0f);
                                    j->addVelocity(impulse/2.0f);
                                }
                            }
                        }
                        cloo++;
                    }
                    ploo++;
                }
            }
            choo++;
        }

        //------------------------------------------POSITION----------------------------------------

        for(int i=0; i<m_num_points; ++i)
        {
            m_particles[i].updatePrevPosition();
            m_particles[i].updatePosition(m_timestep);
        }

        //-----------------------------------------HASH------------------------------

        hashParticles();

        //----------------------------------DOUBLEDENSITY------------------------------------------
        int count =0;

        for(int k = 0; k<(int)m_grid.size(); ++k)
        {
            std::vector<Particle *> neighbours=getSurroundingParticles(count,1,false);

            for(auto& i : m_grid[k])
            {
                float density =0;
                float neardensity=0;
                for(auto& j : neighbours)
                {
                    Vec3 rij = j->getPosition()-i->getPosition();
                    float rijmag = rij.length();
                    float q = rijmag/m_interactionradius;
                    if(q<1 && q!=0) // q==0 when same particle
                    {
                        density+=(1.0f-q)*(1.0f-q);
                        neardensity+=(1.0f-q)*(1.0f-q)*(1.0f-q);
                    }
                }

                float p0 = i->getProperties()->getP0();
                float k = i->getProperties()->getK();
                float knear = i->getProperties()->getKnear();

                float P = k*(density -p0);
                float Pnear = knear * neardensity;
                Vec3 dx = Vec3();
                for(auto& j : neighbours)
                {
                    Vec3 rij = j->getPosition()-i->getPosition();
                    float rijmag = rij.length();
                    float q = rijmag/m_interactionradius;
                    if(q<1 && q!=0)
                    {
                        rij.normalize();
                        Vec3 D = rij*(m_timestep*m_timestep*(P*(1.0f-q))+Pnear*(1.0f-q)*(1.0f-q));
                        j->addPosition(D/2);
                        dx-=(D/2);
                    }
                }
                i->addPosition(dx);
            }
            count++;
        }
        //----------------------------------MAKE NEW VELOCITY-------------------------------------

        for(auto& list : m_grid)
        {
            for(auto& i : list)
            {
                i->setVelocity((i->getPosition()-i->getPrevPosition())/m_timestep);
            }
        }
    }
}

//---------------------------------HASH FUNCTIONS--------------------------------------------------------

void WorldCPU::hashParticles()
{
    int gridSize = m_gridResolution*m_gridResolution;
    std::vector<Particle *> newvector;
    m_grid.assign(gridSize,newvector);
    int grid_cell;
    for(int i=0; i<m_num_points; ++i)
    {
        float positionx = m_particles[i].getPosition()[0];
        float positiony = m_particles[i].getPosition()[1];

        if(positionx<0.0f) positionx=0.0f;
        else if (positionx>1.0f) positionx=1.0f;
        if(positiony<0.0f) positiony=0.0f;
        else if (positiony>1.0f) positiony=1.0f;

        grid_cell=floor(positionx*m_gridResolution)*m_gridResolution + floor(positiony*m_gridResolution);

        if(grid_cell>=0 && grid_cell<gridSize)
        {
            m_grid[grid_cell].push_back(&m_particles[i]);
        }

    }
}

std::vector<Particle *> WorldCPU::getSurroundingParticles(int thiscell, int numsur, bool dragselect) const
{
    int numSurrounding=1;
    std::vector<Particle *> surroundingParticles;

    for(int i = -numSurrounding; i <= numSurrounding; ++i)
    {
        for(int j = -numSurrounding; j <= numSurrounding; ++j)
        {
            int grid_cell = thiscell+ i*m_gridResolution + j;
            if(grid_cell<(m_gridResolution*m_gridResolution) && grid_cell>=0)
            {
                for(auto& p : m_grid[grid_cell])
                {
                    surroundingParticles.push_back(p);
                }
            }
        }
    }

    return surroundingParticles;
}
