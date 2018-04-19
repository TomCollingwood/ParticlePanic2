TEMPLATE = app
TARGET=test
CONFIG += c++11
CONFIG += opengl
INCLUDEPATH += $$INC_INSTALL_DIR ../include
OBJECTS_DIR = obj

QMAKE_CXXFLAGS += -std=c++11 -Wall -Wextra -pedantic

SOURCES += \
    src/Main.cpp

HEADERS +=

LIBS += -L/usr/local/lib -L../lib -L$$LIB_INSTALL_DIR -lPP2_cpu -lPP2_gpu

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

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../libPP2_cpu/release/ -lPP2_cpu
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../libPP2_cpu/debug/ -lPP2_cpu
else:unix: LIBS += -L$$OUT_PWD/../libPP2_cpu/ -lPP2_cpu

INCLUDEPATH += $$PWD/../libPP2_cpu
DEPENDPATH += $$PWD/../libPP2_cpu

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../libPP2_gpu/release/ -lPP2_gpu
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../libPP2_gpu/debug/ -lPP2_gpu
else:unix: LIBS += -L$$OUT_PWD/../libPP2_gpu/ -lPP2_gpu

INCLUDEPATH += $$PWD/../libPP2_gpu
DEPENDPATH += $$PWD/../libPP2_gpu
