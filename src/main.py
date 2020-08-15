from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtOpenGL import *
from sympy.parsing.sympy_parser import parse_expr
import linecache
import os.path
from formulaWindow import *

import OpenGL.GL as gl
from OpenGL import GLU
from OpenGL.GL.shaders import *
import glfw

import numpy as np
import ctypes
from PIL import Image

import time
import sys
import decimal
import random

curFractal = 0
power = 2
isJulia = False
smoothColoring = True

fbo = None

WIDTH = 800
HEIGHT = 600

itr = 200
zoom = 100.0
offsetX = 0.0
offsetY = 0.0

StartX = 0.0
StartY = 0.0

col1R = 10
col1G = 50
col1B = 100
col2R = 70
col2G = 0
col2B = 100
col3R = 89
col3G = 0
col3B = 10
col4R = 100
col4G = 50
col4B = 0
col5R = 100
col5G = 100
col5B = 12
col6R = 100
col6G = 100
col6B = 0
col7R = 50
col7G = 100
col7B = 0
col8R = 0
col8G = 100
col8B = 0
col9R = 0
col9G = 100
col9B = 50
col10R = 0
col10G = 100
col10B = 0

import glm

glfw.init()

decimal.getcontext().prec = 10

c = glm.vec2()
z = glm.vec2(StartX, StartY)

dragging = False
draggingParam = False

zoomInFactor = 2.0
zoomOutFactor = 1 / zoomInFactor

z = glm.dvec2(StartX, StartY)
znew = glm.dvec2()

oldX, oldY = float(), float()

oldStartX, oldStartY = float(), float()

layer = 0.0

i = 0

vertices = [
    -1.0,  1.0, 
    -1.0, -1.0,
     1.0, -1.0,

    -1.0,  1.0,
     1.0, -1.0,
     1.0,  1.0
]

import glm

vertices = np.array(vertices, dtype=ctypes.c_float)

scrVertices = [
    -1.0,  1.0,   0.0,  1.0,
    -1.0, -1.0,   0.0,  0.0,
     1.0, -1.0,   1.0,  0.0,

    -1.0,  1.0,   0.0,  1.0,
     1.0, -1.0,   1.0,  0.0,
     1.0,  1.0,   1.0,  1.0
]

scrVertices = np.array(scrVertices, dtype=np.float32)

vertexShader = """
#version 410 core
layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 texcoord;

out vec2 aTexcoord;

void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    aTexcoord = texcoord;
}"""

fragmentShader = """
#version 410 core

uniform int itr;
uniform double zoom;
uniform dvec2 screenSize;
uniform dvec2 offset;

uniform sampler1D gradient;

uniform float layer;
uniform float time;

double n = 0.0;
double threshold = 100.0;

uniform vec3 col1;
uniform vec3 col2;
uniform vec3 col3;
uniform vec3 col4;
uniform vec3 col5;
uniform vec3 col6;
uniform vec3 col7;
uniform vec3 col8;
uniform vec3 col9;
uniform vec3 col10;

in vec2 aTexcoord;
uniform dvec2 cFormula;

bool escaped = false;

vec3 palette[10] = vec3[10](vec3(col1), vec3(col2), vec3(col3), vec3(col4), vec3(col5), vec3(col6), vec3(col7), vec3(col8), vec3(col9), vec3(col10));

/* 
0 - Mandelbrot
1 - Burning Ship
2 - Tricorn / Mandelbar
*/
dvec2 z;

uniform int FRACTAL_TYPE;
uniform int POWER;

uniform bool juliaEnabled;
uniform bool smoothColoring;

uniform dvec2 Start;

#define twopi 6.283185
#define halfpi twopi / 4

dvec2 c_add(double ar, double ai, double br, double bi, double or, double oi) {
    or = ar + br;
    oi = ai + bi;
    return dvec2(or, oi);
}

dvec2 c_pow(double ar, double ai, int power) {
    dvec2 z = dvec2(ar, ai);
    for (int i = 0; i < power - 1; i++) {
        double or = z.x*ar-z.y*ai;
        double oi = z.x*ai+z.y*ar;
        z = dvec2(or, oi);
    }
    return z;
}

dvec2 c_powcel(double ar, double ai, int power) {
    dvec2 z = dvec2(ar, ai);
    for (int i = 0; i < power - 1; i++) {
        double or = abs(z.x*ar-z.y*ai);
        double oi = z.x*ai+z.y*ar;
        z = dvec2(or, oi);
    }
    return z;
}

dvec2 complex(dvec2 a, dvec2 b) {
    dvec2 ans = dvec2(0.0);
    ans.x = (a.x*b.x)-(a.x*b.x);
    ans.y = (a.y*b.x)+(a.x*b.y);
    return ans;
}

dvec2 z2(dvec2 num) {
    return dvec2(pow(float(num.x),2)-pow(float(num.y),2),2*num.x*num.y);
}

double mandelbrot(dvec2 c) {
    z = vec2(Start.x, Start.y);

    if (smoothColoring) {
        double c2 = dot(c, c);
    }

    const double B = 256.0;
    double l = 0.0;

    for (int i = 0; i < itr; i++) {
        dvec2 znew;

        double zrsqr = z.x*z.x;
        double zisqr = z.y*z.y;
        double zrzrsqr = zrsqr*zrsqr;
        double zizisqr = zisqr*zisqr;
        double zrzisqr = zrsqr*zisqr;

        dvec2 zc = c_pow(z.x, z.y, POWER);
        dvec2 zabsr = c_pow(-abs(z.x), z.y, POWER);
        dvec2 zabsi = c_pow(z.x, -abs(z.y), POWER);
    
        if (FRACTAL_TYPE == 0) {
            znew = zc+c;
        } else if (FRACTAL_TYPE == 1) {
            znew = dvec2(zc.x, -zc.y) + c;
        } else if (FRACTAL_TYPE == 2) {
            znew = dvec2(zc.x, -abs(zc.y)) + c;
        } else if (FRACTAL_TYPE == 3) {
            znew = zabsr + c;
        } else if (FRACTAL_TYPE == 4) {
            znew = zabsi + c;
        } else if (FRACTAL_TYPE == 5) {
            znew = c_powcel(z.x, z.y, POWER) + c;
        } else if (FRACTAL_TYPE == 6) {
            znew = c_powcel(z.x, -z.y, POWER) + c;
        } else if (FRACTAL_TYPE == 7) {
            znew = c_powcel(-abs(z.x), z.y, POWER) + c;
        } else if (FRACTAL_TYPE == 8) {
            znew = c_powcel(abs(z.x), -abs(z.y), POWER) + c;
        } else if (FRACTAL_TYPE == 9) {
            znew = c_powcel(z.x, -abs(z.y), POWER) + c;
        } else if (FRACTAL_TYPE == 10) {
            if (mod(i,4)==0) {
                znew = zc + c;
            }
            if (mod(i,4)==1) {
                znew = dvec2(zc.x, -abs(zc.y)) + c;
            }
            if (mod(i,4)==2) {
                znew = zc + c;
            }
            if (mod(i,4)==3) {
                znew = zc + c;
            }
        } else if (FRACTAL_TYPE == 11) {
            znew = c_pow(abs(z.x), z.y, POWER) + c;
        } else if (FRACTAL_TYPE == 12) {
            znew = c_powcel(abs(z.x), z.y, POWER) + c;
        } else if (FRACTAL_TYPE == 13) {
            znew.x = z.x*z.x - z.y*z.y + c.x;
            znew.y = 2.0 * z.x  * z.y + c.y;
        }

        z = znew;
        if(dot(z, z) > threshold) {
            escaped = true;
            break;
        }
        if (dot(z, z) < threshold && i == itr - 1) {
            return double(0.0);
        }
        n++;
    }

    if (smoothColoring) {
        double sl = n - log2(log2(float(dot(z,z)))) + 4.0;

        double al = smoothstep(-0.1, 0.0, sin(0.5*6.2831));
        n = mix(n, sl, al);
    }

    return n / float(itr);
}

double julia(dvec2 z) {
    dvec2 c = vec2(Start.x, Start.y);
    for (int i = 0; i < itr; i++) {
        dvec2 znew;

        double zrsqr = z.x*z.x;
        double zisqr = z.y*z.y;
        double zrzrsqr = zrsqr*zrsqr;
        double zizisqr = zisqr*zisqr;
        double zrzisqr = zrsqr*zisqr;

        dvec2 zc = c_pow(z.x, z.y, POWER);
        dvec2 zabsr = c_pow(-abs(z.x), z.y, POWER);
        dvec2 zabsi = c_pow(z.x, -abs(z.y), POWER);
    
        if (FRACTAL_TYPE == 0) {
            znew = zc + c;
        } else if (FRACTAL_TYPE == 1) {
            znew = dvec2(zc.x, -zc.y) + c;
        } else if (FRACTAL_TYPE == 2) {
            znew = dvec2(zc.x, -abs(zc.y)) + c;
        } else if (FRACTAL_TYPE == 3) {
            znew = zabsr + c;
        } else if (FRACTAL_TYPE == 4) {
            znew = zabsi + c;
        } else if (FRACTAL_TYPE == 5) {
            znew = c_powcel(z.x, z.y, POWER) + c;
        } else if (FRACTAL_TYPE == 6) {
            znew = c_powcel(z.x, -z.y, POWER) + c;
        } else if (FRACTAL_TYPE == 7) {
            znew = c_powcel(-abs(z.x), z.y, POWER) + c;
        } else if (FRACTAL_TYPE == 8) {
            znew = c_powcel(abs(z.x), -abs(z.y), POWER) + c;
        } else if (FRACTAL_TYPE == 9) {
            znew = c_powcel(z.x, -abs(z.y), POWER) + c;
        } else if (FRACTAL_TYPE == 10) {
            if (mod(i,4)==0) {
                znew = zc + c;
            }
            if (mod(i,4)==1) {
                znew = dvec2(zc.x, -abs(zc.y)) + c;
            }
            if (mod(i,4)==2) {
                znew = zc + c;
            }
            if (mod(i,4)==3) {
                znew = zc + c;
            }
        } else if (FRACTAL_TYPE == 11) {
            znew = c_pow(abs(z.x), z.y, POWER) + c;
        } else if (FRACTAL_TYPE == 12) {
            znew = c_powcel(abs(z.x), z.y, POWER) + c;
        } else if (FRACTAL_TYPE == 13) {
            znew.x = z.x*z.x - z.y*z.y + c.x;
            znew.y = 2.0 * z.x  * z.y + c.y;
        } 

        z = znew;
        if((z.x*z.x)+(z.y*z.y) > threshold) {
            escaped = true;
            break;
        }
        n++;
    }
    return n / float(itr);
}

double mandelbrot3d(dvec2 c, float zlayer) {
    dvec3 z = vec3(Start.x, Start.y, zlayer);
    for (int i = 0; i < itr; i++) {
        dvec3 znew;

        znew.x = (z.x*z.x)-(z.y*z.y)-(z.z*z.z)+c.x;
        znew.y = (2.0*z.x*z.y)+c.y;
        znew.z = (-2.0*z.x*zlayer);

        z = znew;

        if ((z.x*z.x)+(z.y*z.y)+(z.z*z.z)>threshold) break;
        n++;
    }

    return n / float(itr);
}

float x = 1.0 / 9.0;

vec3 mapToColor(float t, vec3 c1, vec3 c2, vec3 c3, vec3 c4, vec3 c5, vec3 c6, vec3 c7, vec3 c8, vec3 c9, vec3 c10) {
    if (t < x) return mix(c1, c2, t/x);
    else if (t < 2.0 * x) return mix(c2, c3, (t - x)/x);
    else if (t < 3.0 * x) return mix(c3, c4, (t - 2.0*x)/x);
    else if (t < 4.0 * x) return mix(c4, c5, (t - 3.0*x)/x);
    else if (t < 5.0 * x) return mix(c5, c6, (t - 4.0*x)/x);
    else if (t < 6.0 * x) return mix(c6, c7, (t - 5.0*x)/x);
    else if (t < 7.0 * x) return mix(c7, c8, (t - 6.0*x)/x);
    else if (t < 8.0 * x) return mix(c8, c9, (t - 7.0*x)/x);
    else if (t < 9.0 * x) return mix(c9, c10, (t - 8.0*x)/x);

    return c10;
}

void main() {
    dvec2 coord = dvec2(gl_FragCoord.xy);
    vec2 resolution = vec2(3.0, 3.0);
    vec2 direction = vec2(1.0, 0.5);
    vec2 off1 = vec2(1.3846153846) * direction;
    vec2 off2 = vec2(3.2307692308) * direction;
    dvec2 z = vec2(Start.x, Start.y);
    dvec2 c = ((coord - screenSize / 2) / zoom) - offset;
    dvec2 znew;

    double t;
    if (!juliaEnabled)
        t = mandelbrot(((coord - screenSize / 2) / zoom) - offset);
    else
        t = julia(((coord - screenSize / 2) / zoom) - offset);

    //double t3d;
    //t3d = mandelbrot3d(((coord - screenSize / 2) / zoom) - offset, 1.0);

    if (gl_FragCoord.x < 40)
        gl_FragColor = vec4(1.0);

    vec4 color = escaped ? vec4(mapToColor(float(mod(t, itr)), vec3(palette[0]), vec3(palette[1]), vec3(palette[2]), vec3(palette[3]), vec3(palette[4]), vec3(palette[5]), vec3(palette[6]), vec3(palette[7]), vec3(palette[8]), vec3(palette[9])), 1.0) : vec4(vec3(0.0), 1.0);

    gl_FragColor = color;
}
"""

screenVert = """
#version 410 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main() {
    TexCoords = aTexCoords;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

screenFragment = """
#version 410 core
in vec2 TexCoords;

uniform sampler2D tex;
uniform float time;

float hash(vec2 x)
{
	return fract(cos(dot(x.xy,vec2(2.31,53.21))*124.123)*412.0); 
}

void main() {
    vec3 result = texture2D(tex, TexCoords).xyz;

    float acc = .0;

    for (int smp = 0; smp < 16; smp++) {
        float tnoise = hash(TexCoords.xy+vec2(smp,smp))*(1.0/24.0);

        vec2 temp = TexCoords-result.xy+vec2(.0,3.0+4.0);
        float s = sqrt(dot(temp, temp));
        s -= 1.0;
        s *= gl_FragCoord.y*.05;
        s = min(1.0,max(.0,s));

        acc += s/float(16);
    }

    result += hash(TexCoords.xy+result.xy)*.02;

    gl_FragColor = vec4(result, 1.0);
}
"""

program = None
scrProgram = None

postProcessing = False

def loadTexture(path):
    img = Image.open(path)
    img_data = np.frombuffer(img.tobytes(), np.uint8)

    width = img.width

    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_1D, texture)
    gl.glTexParameterf(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameterf(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST_MIPMAP_NEAREST)
    gl.glTexImage1D(gl.GL_TEXTURE_1D, 0, gl.GL_RGBA, width, 0,
        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, img_data)
    gl.glGenerateMipmap(gl.GL_TEXTURE_1D)
    return texture

def initFramebuffer():
    pass

class GLWidget(QGLWidget):
    global zoom
    global program
    global offsetX
    global offsetY
    global col1R
    global col1G
    global col1B
    global texture
    global fbo

    def __init__(self, parent=None):
        self.parent = parent
        QGLWidget.__init__(self, parent)
        self.lastDragPos = QPoint()
        self.setCursor(Qt.CrossCursor)

    def initializeGL(self):
        self.qglClearColor(QColor(255, 0, 0))
    
    def resizeGL(self, width, height):
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / float(height)

        GLU.gluPerspective(45.0, aspect, 1.0, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self):
        deltaTime = 0.0
        lastFrame = 0.0
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 8, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)

        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        scrVao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(scrVao)
        scrVbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, scrVbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, scrVertices.nbytes, scrVertices, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, False, 16, ctypes.c_void_p(0))
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, False, 16, ctypes.c_void_p(8))
        gl.glEnableVertexAttribArray(0)
        gl.glEnableVertexAttribArray(1)

        fbo = gl.glGenFramebuffers(1)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        renderedTexture = gl.glGenTextures(1)

        gl.glBindTexture(gl.GL_TEXTURE_2D, renderedTexture)

        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, 550, 429, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        drb = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, drb)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, 550, 429)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, renderedTexture, 0)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, drb)

        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer binding failed")

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)

        currentFrame = glfw.get_time()
        deltaTime = currentFrame - lastFrame
        lastFrame = currentFrame

        if (postProcessing):
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0., 0., 0., 1.0)
        gl.glClearDepth(1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))
        scrProgram = compileProgram(compileShader(screenVert, gl.GL_VERTEX_SHADER), compileShader(screenFragment, gl.GL_FRAGMENT_SHADER))

        gl.glBindVertexArray(vao)
        gl.glUseProgram(program)

        gl.glUniform2d(gl.glGetUniformLocation(program, "screenSize"), 550, 420)
        gl.glUniform2d(gl.glGetUniformLocation(program, "offset"), offsetX, offsetY)
        gl.glUniform1d(gl.glGetUniformLocation(program, "zoom"), zoom)
        gl.glUniform1i(gl.glGetUniformLocation(program, "itr"), itr)
        gl.glUniform1i(gl.glGetUniformLocation(program, "FRACTAL_TYPE"), curFractal)
        gl.glUniform1i(gl.glGetUniformLocation(program, "POWER"), power)
        gl.glUniform2d(gl.glGetUniformLocation(program, "Start"), StartX, StartY)
        gl.glUniform1f(gl.glGetUniformLocation(program, "juliaEnabled"), isJulia)
        # gl.glUniform1i(gl.glGetUniformLocation(program, "gradient"), 0)
        gl.glUniform1f(gl.glGetUniformLocation(program, "layer"), layer)
        gl.glUniform2d(gl.glGetUniformLocation(program, "cFormula"), z.x*z.x-z.y*z.y, 2.0*abs(z.x)*z.y)
        gl.glUniform1f(gl.glGetUniformLocation(program, "smoothColoring"), smoothColoring)
        gl.glUniform1f(gl.glGetUniformLocation(program, "time"), glfw.get_time())

        gl.glUniform3f(gl.glGetUniformLocation(program, "col1"), col1R / 100, col1G / 100, col1B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col2"), col2R / 100, col2G / 100, col2B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col3"), col3R / 100, col3G / 100, col3B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col4"), col4R / 100, col4G / 100, col4B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col5"), col5R / 100, col5G / 100, col5B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col6"), col6R / 100, col6G / 100, col6B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col7"), col7R / 100, col7G / 100, col7B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col8"), col8R / 100, col8G / 100, col8B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col9"), col9R / 100, col9G / 100, col9B / 100)

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        if (postProcessing):
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

            gl.glClearColor(1., 0., 0., 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glDisable(gl.GL_DEPTH_TEST)

            gl.glBindVertexArray(scrVao)
            gl.glUseProgram(scrProgram)

            self.update()

            gl.glUniform1f(gl.glGetUniformLocation(scrProgram, "time"), glfw.get_time())
            gl.glBindTexture(gl.GL_TEXTURE_2D, renderedTexture)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glBindVertexArray(0)

    def wheelEvent(self, event):
        global zoom
        global program
        global offsetX
        global offsetY
        global itr

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        dx = (event.pos().x() - 550 / 2) / zoom - offsetX
        dy = (420 - event.pos().y() - 420 / 2) / zoom - offsetY

        offsetX = -dx
        offsetY = -dy
        if event.angleDelta().y() < 0:
            zoom /= 1.3
        else:
            zoom *= 1.3

        if zoom <= 100.0:
            zoom = 100.0

        dx = (event.pos().x() - 550 / 2) / zoom
        dy = (420 - event.pos().y() - 420 / 2) / zoom
        offsetX += dx
        offsetY += dy

        self.update()

        gl.glUniform1d(gl.glGetUniformLocation(program, "zoom"), zoom)
        gl.glUniform2d(gl.glGetUniformLocation(program, "offset"), offsetX, offsetY)
        gl.glUniform1i(gl.glGetUniformLocation(program, "itr"), itr)

    def mousePressEvent(self, event):
        global dragging
        global oldX
        global oldY
        global oldStartX
        global oldStartY
        global offsetX
        global offsetY
        global zoom
        global juliaTrigger
        global draggingParam

        if event.button() == Qt.LeftButton:
            dragging = True
            draggingParam = False

            oldX = event.pos().x()
            oldY = event.pos().y()

        if event.button() == Qt.RightButton and juliaTrigger.isChecked():
            draggingParam = True
            dragging = False

            oldStartX = event.pos().x()
            oldStartY = event.pos().y()

    def mouseMoveEvent(self, event):
        global dragging
        global offsetX
        global offsetY
        global oldX
        global oldY
        global oldStartX
        global oldStartY
        global program
        global zoom
        global StartX
        global StartY
        global juliaTrigger

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        self.update()

        if dragging:
            offsetX += (event.pos().x() - oldX) / zoom
            offsetY += (oldY - event.pos().y()) / zoom

            oldX = event.pos().x()
            oldY = event.pos().y()

            gl.glUniform2d(gl.glGetUniformLocation(program, "offset"), offsetX, offsetY)

        if draggingParam and juliaTrigger.isChecked():
            StartX += (event.pos().x() - oldStartX) / zoom
            StartY += (oldStartY - event.pos().y()) / zoom

            oldStartX = event.pos().x()
            oldStartY = event.pos().y()

            StartRe.setText(str(StartX))
            StartIm.setText(str(StartY))

            gl.glUniform2d(gl.glGetUniformLocation(program, "Start"), StartX, StartY)

    def mouseReleaseEvent(self, event):
        global dragging
        global offsetX
        global offsetY
        global oldX
        global oldY
        global oldStartX
        global oldStartY
        global StartX
        global StartY
        global juliaTrigger
        global StartRe
        global StartIm

        if event.button() == Qt.LeftButton:
            offsetX += event.pos().x() - oldX
            offsetY += oldY - event.pos().y()

            dragging = False
            draggingParam = False

        if event.button() == Qt.RightButton and juliaTrigger.isChecked():

            dragging = False
            draggingParam = False

    def zoomIn(self, zoomFactor):
        global zoom
        global program

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        zoom *= zoomFactor

        self.update()

        gl.glUniform1d(gl.glGetUniformLocation(program, "zoom"), zoom)

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.resize(WIDTH + 20, 490)
        self.setWindowTitle("UniFract++")

        self.fractalEditor = fractalEditor(self)

        self.glWidget = GLWidget(self)
        self.glWidget.resize(550, 420)
        self.glWidget.move(5, 65)

        self.initUI()

    def initUI(self):
        global fractalType
        global program
        global power
        global Power
        global StartRe
        global StartIm
        global iterations
        global itr
        global Zoom
        global juliaTrigger
        global gradientBtn
        global R1
        global B1
        global G1
        global R2
        global B2
        global G2
        global R3
        global B3
        global G3
        global R4
        global B4
        global G4
        global zlay
        global formulaBtn

        # Tabs

        tabs = QTabWidget(self)
        tab1 = QWidget(self)
        tab2 = QWidget(self)
        tab3 = QWidget(self)
        tabs.resize(255, 420)

        tabs.move(560, self.glWidget.y())

        tabs.addTab(tab1, "Fractal")
        tabs.addTab(tab2, "Rendering")
        tabs.addTab(tab3, "Gradient")

        tab1.layout = QFormLayout()
        tab2.layout = QFormLayout()
        tab3.layout = QFormLayout()

        # Main Menu

        mainMenu = QMenuBar(self)
        fileMenu = mainMenu.addMenu("&File")
        editMenu = mainMenu.addMenu("&Edit")

        # Tool Bar

        openCfg = QPushButton(self)
        openCfg.setText("Open Config")
        openCfg.move(5, 30)

        saveCfg = QPushButton(self)
        saveCfg.setText("Save Config")
        saveCfg.move(openCfg.width() + 10, 30)

        exportIm = QPushButton(self)
        exportIm.setText("Export Image")
        exportIm.move(openCfg.width() * 2 + 20, 30)

        gradientBtn = QPushButton(self)
        gradientBtn.setText("Gradient")
        gradientBtn.move(openCfg.width() * 3 + 30, 30)
        gradientBtn.clicked.connect(self.editGradient)

        initialState = QPushButton(self)
        initialState.setText("Initial State")
        initialState.move(openCfg.width() * 4 + 40, 30)
        initialState.clicked.connect(self.resetEverything)

        # Tab 1 (Fractal)

        fractalTypes = ["Mandelbrot", "Tricorn", "Burning Ship", "Perpendicular Mandelbrot", "Perpendicular Burning Ship", "Celtic Mandelbrot", "Celtic Mandelbar", "Perpendicular Celtic", "Buffalo", "Perpendicular Buffalo", "Mandelship", "Mandelbrot Heart", "Buffalo Heart", "Custom"]

        fractalType = QComboBox()
        fractalType.addItems(fractalTypes)

        fractalType.currentIndexChanged.connect(self.changeFractal)

        Power = QSpinBox()
        Power.setValue(2)
        Power.setMinimum(2)
        Power.setMaximum(5)
        Power.valueChanged.connect(self.changePower)

        StartingP = QHBoxLayout()
        StartRe = QLineEdit()
        StartRe.setValidator(QDoubleValidator())
        StartRe.setPlaceholderText("Real")
        StartRe.setText(str(StartX))
        StartIm = QLineEdit()
        StartIm.setValidator(QDoubleValidator())
        StartIm.setPlaceholderText("Imag")
        StartIm.setText(str(StartY))
        StartingP.addWidget(StartRe)
        StartingP.addWidget(StartIm)

        StartRe.textChanged.connect(self.changeStartP)
        StartIm.textChanged.connect(self.changeStartP)

        iterations = QLineEdit()
        iterations.setValidator(QIntValidator())
        iterations.setText(str(itr))
        iterations.textChanged.connect(self.changeIter)

        Zoom = QLineEdit()
        Zoom.setValidator(QDoubleValidator())

        self.timer = QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.info)
        self.timer.start()

        juliaTrigger = QCheckBox()
        juliaTrigger.setChecked(False)
        juliaTrigger.clicked.connect(self.setJulia)

        zlay = QSlider(Qt.Horizontal)
        zlay.setMinimum(-10)
        zlay.setMaximum(10)
        zlay.valueChanged.connect(self.changeLayer)

        formulaBtn = QPushButton()
        formulaBtn.setText("Edit Custom Formula")
        formulaBtn.clicked.connect(self.openWindow)

        if curFractal != 13:
            formulaBtn.setEnabled(False)

        formulaBtn.setEnabled(False)

        tab1.layout.addRow("Fractal", fractalType)
        tab1.layout.addRow("Power", Power)
        tab1.layout.addRow("Seed", StartingP)
        tab1.layout.addRow("Iterations", iterations)
        tab1.layout.addRow("Zoom", Zoom)
        tab1.layout.addRow("Julia Set", juliaTrigger)
        tab1.layout.addRow(formulaBtn)
        # tab1.layout.addRow("Z Layer", zlay)

        tab1.setLayout(tab1.layout)

        # Tab 2 (Rendering)

        precision = QComboBox()
        precision.addItem("Double")

        precision.setDisabled(True)

        itrCount = QComboBox()
        itrCount.addItem("Smooth iteration")
        itrCount.addItem("Discrete iteration")

        tab2.layout.addRow("Float Type", precision)
        tab2.layout.addRow("Iteration count", itrCount)

        tab2.setLayout(tab2.layout)

        # Tab 3 (Gradient)

        col1 = QHBoxLayout()
        col2 = QHBoxLayout()
        col3 = QHBoxLayout()
        col4 = QHBoxLayout()

        R1 = QSlider(Qt.Horizontal)
        R1.setMaximum(100)
        R1.setMinimum(0)

        R1.valueChanged.connect(self.editGradient)

        G1 = QSlider(Qt.Horizontal)
        G1.setMaximum(100)
        G1.setMinimum(0)

        G1.valueChanged.connect(self.editGradient)

        B1 = QSlider(Qt.Horizontal)
        B1.setMaximum(100)
        B1.setMinimum(0)

        B1.valueChanged.connect(self.editGradient)

        R1.setValue(col1R)
        G1.setValue(col1G)
        B1.setValue(col1B)

        col1.addWidget(R1)
        col1.addWidget(G1)
        col1.addWidget(B1)

        R2 = QSlider(Qt.Horizontal)
        R2.setMaximum(100)
        R2.setMinimum(0)

        R2.valueChanged.connect(self.editGradient)

        G2 = QSlider(Qt.Horizontal)
        G2.setMaximum(100)
        G2.setMinimum(0)

        G2.valueChanged.connect(self.editGradient)

        B2 = QSlider(Qt.Horizontal)
        B2.setMaximum(100)
        B2.setMinimum(0)

        B2.valueChanged.connect(self.editGradient)

        R2.setValue(col2R)
        G2.setValue(col2G)
        B2.setValue(col2B)

        col2.addWidget(R2)
        col2.addWidget(G2)
        col2.addWidget(B2)

        R3 = QSlider(Qt.Horizontal)
        R3.setMaximum(100)
        R3.setMinimum(0)

        R3.valueChanged.connect(self.editGradient)

        G3 = QSlider(Qt.Horizontal)
        G3.setMaximum(100)
        G3.setMinimum(0)

        G3.valueChanged.connect(self.editGradient)

        B3 = QSlider(Qt.Horizontal)
        B3.setMaximum(100)
        B3.setMinimum(0)

        B3.valueChanged.connect(self.editGradient)

        R3.setValue(col3R)
        G3.setValue(col3G)
        B3.setValue(col3B)

        col3.addWidget(R3)
        col3.addWidget(G3)
        col3.addWidget(B3)

        R4 = QSlider(Qt.Horizontal)
        R4.setMaximum(100)
        R4.setMinimum(0)

        R4.valueChanged.connect(self.editGradient)

        G4 = QSlider(Qt.Horizontal)
        G4.setMaximum(100)
        G4.setMinimum(0)

        G4.valueChanged.connect(self.editGradient)

        B4 = QSlider(Qt.Horizontal)
        B4.setMaximum(100)
        B4.setMinimum(0)

        B4.valueChanged.connect(self.editGradient)

        R4.setValue(col4R)
        G4.setValue(col4G)
        B4.setValue(col4B)

        col4.addWidget(R4)
        col4.addWidget(G4)
        col4.addWidget(B4)

        # Random Colors

        randCol = QPushButton()
        randCol.setText("Random Colors")
        randCol.clicked.connect(self.randomColors)

        # Interior Color

        inColorBtn = QPushButton('Interior Color')
        inColorBtn.clicked.connect(self.inColor)

        tab3.layout.addRow("Color 1", col1)
        tab3.layout.addRow("Color 2", col2)
        tab3.layout.addRow("Color 3", col3)
        tab3.layout.addRow("Color 4", col4)
        tab3.layout.addRow(randCol)
        tab3.layout.addRow(inColorBtn)

        tab3.setLayout(tab3.layout)

    def inColor(self):
        color = QColorDialog.getColor()

        if color.isValid():
            print(color.getRgb())

    def openWindow(self):
        self.fractalEditor.show()
        self.glWidget.update()

    def changeFractal(self):
        global fractalType
        global program
        global curFractal
        global formulaBtn

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        curFractal = fractalType.currentIndex()

        self.glWidget.update()

        if curFractal == 13:
            formulaBtn.setEnabled(True)
        else:
            formulaBtn.setEnabled(False)

        gl.glUniform1i(gl.glGetUniformLocation(program, "FRACTAL_TYPE"), curFractal)

    def setJulia(self):
        global juliaTrigger
        global program
        global isJulia
        global StartY
        global StartX

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        self.glWidget.update()

        isJulia = juliaTrigger.isChecked()

        if juliaTrigger.isChecked():
            StartX = -1.0
            StartY = -0.5
        else:
            StartX, StartY = 0.0, 0.0

        gl.glUniform1f(gl.glGetUniformLocation(program, "juliaEnabled"), isJulia)

    def changePower(self):
        global power
        global Power

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        self.glWidget.update()

        power = Power.value()
        gl.glUniform1i(gl.glGetUniformLocation(program, "POWER"), power)

    def changeLayer(self):
        global zlay
        global program
        global layer

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        self.glWidget.update()

        layer = zlay.value() / 10

        gl.glUniform1f(gl.glGetUniformLocation(program, "layer"), layer)

    def changeStartP(self):
        global StartX
        global StartY
        global StartRe
        global StartIm
        global program

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        self.glWidget.update()

        if StartRe.text() != "" and StartIm.text() != "":
            StartX = float(StartRe.text())
            StartY = float(StartIm.text())

            gl.glUniform2d(gl.glGetUniformLocation(program, "Start"), StartX, StartY)

    def changeIter(self):
        global iterations
        global itr
        global program

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        itr = int(iterations.text())

        gl.glUniform1i(gl.glGetUniformLocation(program, "itr"), itr)

        self.glWidget.update()

    def info(self):
        global Zoom
        global zoom

        Zoom.setText(str(zoom))

    def editGradient(self):
        global program
        global R1
        global B1
        global G1
        global R2
        global B2
        global G2
        global R3
        global B3
        global G3
        global R4
        global B4
        global G4
        global col1R
        global col1G
        global col1B
        global col2R
        global col2G
        global col2B
        global col3R
        global col3G
        global col3B
        global col4R
        global col4G
        global col4B

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        col1R = int(R1.value())
        col1G = int(G1.value())
        col1B = int(B1.value())
        col2R = int(R2.value())
        col2G = int(G2.value())
        col2B = int(B2.value())
        col3R = int(R3.value())
        col3G = int(G3.value())
        col3B = int(B3.value())
        col4R = int(R4.value())
        col4G = int(G4.value())
        col4B = int(B4.value())

        gl.glUniform3f(gl.glGetUniformLocation(program, "col1"), col1R / 100, col1G / 100, col1B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col2"), col2R / 100, col2G / 100, col2B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col3"), col3R / 100, col3G / 100, col3B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col4"), col4R / 100, col4G / 100, col4B / 100)

        self.glWidget.update()

    def randomColors(self):
        global col1R
        global col1G
        global col1B
        global col2R
        global col2G
        global col2B
        global col3R
        global col3G
        global col3B
        global col4R
        global col4G
        global col4B
        global col5R
        global col5G
        global col5B
        global col6R
        global col6G
        global col6B
        global col7R
        global col7G
        global col7B
        global col8R
        global col8G
        global col8B
        global col9R
        global col9G
        global col9B
        global col10R
        global col10G
        global col10B
        global program

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        col1R = random.randint(0, 100)
        col1B = random.randint(0, 100)
        col1B = random.randint(0, 100)
        col2R = random.randint(0, 100)
        col2G = random.randint(0, 100)
        col2B = random.randint(0, 100)
        col3R = random.randint(0, 100)
        col3G = random.randint(0, 100)
        col3B = random.randint(0, 100)
        col4R = random.randint(0, 100)
        col4G = random.randint(0, 100)
        col4B = random.randint(0, 100)
        col5R = random.randint(0, 100)
        col5G = random.randint(0, 100)
        col5B = random.randint(0, 100)
        col6R = random.randint(0, 100)
        col6G = random.randint(0, 100)
        col6B = random.randint(0, 100)
        col7R = random.randint(0, 100)
        col7G = random.randint(0, 100)
        col7B = random.randint(0, 100)
        col8R = random.randint(0, 100)
        col8G = random.randint(0, 100)
        col8B = random.randint(0, 100)
        col9R = random.randint(0, 100)
        col9G = random.randint(0, 100)
        col9B = random.randint(0, 100)
        col10R = random.randint(0, 100)
        col10G = random.randint(0, 100)
        col10B = random.randint(0, 100)

        gl.glUniform3f(gl.glGetUniformLocation(program, "col1"), col1R / 100, col1G / 100, col1B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col2"), col2R / 100, col2G / 100, col2B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col3"), col3R / 100, col3G / 100, col3B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col4"), col4R / 100, col4G / 100, col4B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col5"), col5R / 100, col5G / 100, col5B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col6"), col6R / 100, col6G / 100, col6B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col7"), col7R / 100, col7G / 100, col7B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col8"), col8R / 100, col8G / 100, col8B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col9"), col9R / 100, col9G / 100, col9B / 100)

        self.glWidget.update()

    def resetEverything(self):
        global Power
        global StartRe
        global StartIm
        global offsetX
        global offsetY
        global zoom
        global itr
        global juliaTrigger
        global isJulia

        global fractalType

        global program

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        Power.setValue(2)
        StartRe.setText("0.0")
        StartIm.setText("0.0")
        fractalType.setCurrentIndex(0)
        offsetX = 0.0
        offsetY = 0.0
        zoom = 100.0
        itr = 200
        juliaTrigger.setChecked(False)
        isJulia = False

        self.glWidget.update()

        gl.glUniform2d(gl.glGetUniformLocation(program, "offset"), offsetX, offsetY)
        gl.glUniform1d(gl.glGetUniformLocation(program, "zoom"), zoom)
        gl.glUniform2d(gl.glGetUniformLocation(program, "Start"), StartX, StartY)
        gl.glUniform1i(gl.glGetUniformLocation(program, "itr"), itr)

if __name__ == '__main__':

    app = QApplication(sys.argv)

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())
