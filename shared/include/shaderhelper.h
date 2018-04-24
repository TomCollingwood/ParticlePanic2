#ifndef SHADERHELPER_H
#define SHADERHELPER_H

#include "glinc.h"

#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
using namespace std;

/**
  * Static function to read text from a file in a relatively safe manner.
  * \return A big char* of the text, or NULL if it was unsuccessful.
  * \author Richard Southern
  * \date 3/10/2012
  */
static char* textFileRead(const char *fileName) {
    char* text = NULL;

    if (fileName != NULL) {
        FILE *file = fopen(fileName, "rt");

        if (file != NULL) {
            fseek(file, 0, SEEK_END);
            int count = ftell(file);
            rewind(file);


            if (count > 0) {
                text = (char*)malloc(sizeof(char) * (count + 1));
                count = fread(text, sizeof(char), count, file);
                text[count] = '\0';
            }
            fclose(file);
        }
    }
    return text;
}

/**
  * Test the shader works.
  * \return true if it's ok.
  */
static bool validateShader(GLuint shader, const char* file) {
    const unsigned int BUFFER_SIZE = 512;
    char buffer[BUFFER_SIZE];
    memset(buffer, 0, BUFFER_SIZE);
    GLsizei length = 0;

    glGetShaderInfoLog(shader, BUFFER_SIZE, &length, buffer);
    if (length > 0) {
        cerr << "Shader " << shader << " (" << (file?file:"") << ") Compile Message: " << buffer << endl;
    }
    return true;
}

/**
  * Test the program to ensure it works ok.
  * \return true if it's ok.
  */
static bool validateProgram(GLuint program) {
    const unsigned int BUFFER_SIZE = 512;
    char buffer[BUFFER_SIZE];
    memset(buffer, 0, BUFFER_SIZE);
    GLsizei length = 0;

    memset(buffer, 0, BUFFER_SIZE);
    glGetProgramInfoLog(program, BUFFER_SIZE, &length, buffer);
    if (length > 0) {
        cerr << "Program " << program << " Link Message: " << buffer << endl;
    }

    glValidateProgram(program);
    GLint status;
    glGetProgramiv(program, GL_VALIDATE_STATUS, &status);
    if (status == GL_FALSE) {
        cerr << "Program " << program << " Validation Failed" << endl;
        return false;
    }
    return true;
}


#endif // SHADERHELPER_H
