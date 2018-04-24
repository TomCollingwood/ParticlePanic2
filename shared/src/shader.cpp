#include "shader.h"
#include "shaderhelper.h"
#include <string.h>
#include <iostream>
#include <stdlib.h>
//#include <glm/gtx/transform.hpp>

using namespace std;

/**
  * Construct an empty shader.
  */
Shader::Shader() : m_init(false), m_id(0), m_vp(0), m_fp(0) {
}

/**
  * Construct the shader from the filenames of the vertex and then the fragment shader.
  * \param vsFile The name of the input vertex shader file
  * \param fsFile The name of the input fragment shader file
  */
Shader::Shader(const char *vsFile, const char *fsFile) : m_init(false), m_id(0), m_vp(0), m_fp(0) {
    init(vsFile, fsFile);
}

/**
  * Initialise the shader from the filenames of the vertex and then the fragment shader.
  * \param vsFile The name of the input vertex shader file
  * \param fsFile The name of the input fragment shader file
  */
void Shader::init(const char *vsFile, const char *fsFile) {
    if (m_init) kill();

    m_vp = glCreateShader(GL_VERTEX_SHADER);
    m_fp = glCreateShader(GL_FRAGMENT_SHADER);
    
    const char* vsText = textFileRead(vsFile);
    const char* fsText = textFileRead(fsFile);    
    
    if (vsText == NULL || fsText == NULL) {
        cerr << "Either vertex shader or fragment shader file not found." << endl;
        return;
    }
    
    glShaderSource(m_vp, 1, &vsText, 0);
    glShaderSource(m_fp, 1, &fsText, 0);
    
    glCompileShader(m_vp);
    if (!validateShader(m_vp, vsFile)) {
        return;
    }
    glCompileShader(m_fp);
    if (!validateShader(m_fp, fsFile)) {
        return;
    }
    m_id = glCreateProgram();
    glAttachShader(m_id, m_fp);
    glAttachShader(m_id, m_vp);
    glLinkProgram(m_id);
    if (!validateProgram(m_id)) return;

    // If we've got to this stage, we think it is good to go
    m_init = true;
}

/**
  * Destroy the shader by detaching them and deleting the relevant programs.
  */
Shader::~Shader() {
    kill();
}

void Shader::kill() {
    if (m_init) {
        glDetachShader(m_id, m_fp);
        glDetachShader(m_id, m_vp);

        glDeleteShader(m_fp);
        glDeleteShader(m_vp);
        glDeleteProgram(m_id);

        m_id = 0; m_vp = 0; m_fp = 0;
        m_init = false;
    }
}

/**
  * \return the id of the shader (for what purpose, we know not).
  */
GLuint Shader::id() const {
    return m_id;
}

/**
  * Bind (i.e. activate) the shader to whatever render calls follow.
  */
void Shader::bind() const {
    if (m_init) glUseProgram(m_id);
}

/**
  * Unbind (deactivate) the shader, reverting to the traditional fixed function pipeline. Note that this should
  * probably revert to whatever shader was used previously.
  */
void Shader::unbind() const {
    if (m_init) glUseProgram(0);
}

void Shader::printProperties() const {
    GLint nUniforms, nAttribs;
    glGetProgramiv(id(), GL_ACTIVE_UNIFORMS, &nUniforms);
    char name[256]; GLsizei l; int i;
    for (i=0; i<nUniforms; ++i) {
        glGetActiveUniformName(id(), i, 256, &l, name);
        std::cerr << "Uniform"<<i<<":\""<<name<<"\"\n";
    }
    glGetProgramiv(id(), GL_ACTIVE_ATTRIBUTES, &nAttribs);
    GLint size; GLenum type;
    for (i=0; i < nAttribs; ++i) {
        glGetActiveAttrib(id(), i, 256, &l, &size, &type, name);
        std::cerr << "Attribute"<<i<<":\""<<name<<"\" Size:"<<size<<" Type:";
        switch(type) {
        case GL_FLOAT: std::cerr << "GL_FLOAT\n"; break;
        case GL_FLOAT_VEC2: std::cerr << "GL_FLOAT_VEC2\n"; break;
        case GL_FLOAT_VEC3: std::cerr << "GL_FLOAT_VEC3\n"; break;
        case GL_FLOAT_VEC4: std::cerr << "GL_FLOAT_VEC4\n"; break;
        case GL_FLOAT_MAT2: std::cerr << "GL_FLOAT_MAT2\n"; break;
        case GL_FLOAT_MAT3: std::cerr << "GL_FLOAT_MAT3\n"; break;
        case GL_FLOAT_MAT4: std::cerr << "GL_FLOAT_MAT4\n"; break;
        case GL_FLOAT_MAT2x3: std::cerr << "GL_FLOAT_MAT2x3\n"; break;
        case GL_FLOAT_MAT2x4: std::cerr << "GL_FLOAT_MAT2x4\n"; break;
        case GL_FLOAT_MAT3x2: std::cerr << "GL_FLOAT_MAT3x2\n"; break;
        case GL_FLOAT_MAT3x4: std::cerr << "GL_FLOAT_MAT3x4\n"; break;
        case GL_FLOAT_MAT4x2: std::cerr << "GL_FLOAT_MAT4x2\n"; break;
        case GL_FLOAT_MAT4x3: std::cerr << "GL_FLOAT_MAT4x3\n"; break;
        default: std::cerr << "UNKNOWN\n";
        }
    }
}

