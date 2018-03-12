TEMPLATE = app
TARGET=test
CONFIG += c++11
CONFIG += opengl
INCLUDEPATH += $$INC_INSTALL_DIR ../libPP2_cpu/include
OBJECTS_DIR = obj

QMAKE_CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic

SOURCES += \
    src/Toolbar.cpp \
    src/Main.cpp

HEADERS += \
    include/Toolbar.h \
    include/Commands.h

LIBS += -L/usr/local/lib -L../lib -lPP2_cpu -lPP2_gpu

QMAKE_RPATHDIR += $$LIB_INSTALL_DIR

linux: {
  LIBS+=$$system(sdl2-config --libs)
  LIBS += -lSDL2  -lGLU -lGL -lSDL2_image  -L/usr/local/lib/ #-lglut #-lGLEW
}

macx: {
  DEFINES+=MAC_OS_X_VERSION_MIN_REQUIRED=1060
  QMAKE_LFLAGS += -F/Library/Frameworks
  LIBS += -framework SDL2
  LIBS += -framework SDL2_image
  INCLUDEPATH += /Library/Frameworks/SLD2_image.framework
  INCLUDEPATH += /Library/Frameworks/SDL2.framework/Headers
  INCLUDEPATH += /usr/local/include
  INCLUDEPATH += /usr/local/Cellar

  LIBS+= -framework OpenGL
  LIBS+= -framework GLUT
}
