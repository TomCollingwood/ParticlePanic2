#ifndef GLINC_H
#define GLINC_H

#if ( (defined(__MACH__)) && (defined(__APPLE__)) )
#include <stdlib.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#elif (defined(WIN32))
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/gl.h>
#else
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#endif

#endif // GLINC_H
