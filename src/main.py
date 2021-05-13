import time
from OpenGL.error import NullFunctionError
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtOpenGL import *
from formulaWindow import *
from exportImgSeriesDialog import *
from errWindow import *

import OpenGL.GL as gl
from OpenGL.GL import glCreateShader
from OpenGL.GL.ARB.shader_objects import *
from OpenGL.GL.ARB.fragment_shader import *
from OpenGL.GL.ARB.vertex_shader import *
from OpenGL import GLU
from OpenGL.GL.shaders import *
from OpenGL.extensions import alternate
import glfw

#glCreateShader = alternate("glCreateShader", glCreateShader, glCreateShaderObjectARB)

import numpy as np
import ctypes
from PIL import Image

import sys
import decimal
import random

from math import *

import threading

decimal.getcontext().prec = 2

curFractal = 0
power = 2
escRad = 128.0
isJulia = False
perturbationEnabled = False
smoothColoring = True

fbo = None

WIDTH = 800
HEIGHT = 600

itr = 200
zoom = 100.0
offsetX = 0.0
offsetY = 0.0
offsetX128 = [0.0, 0.0]
offsetY128 = [0.0, 0.0]

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

inColorR = 0
inColorG = 0
inColorB = 0

import glm

glfw.init()

decimal.getcontext().prec = 1

useQuadPrec = False

c = glm.vec2()
z = glm.vec2(StartX, StartY)

dragging = False
draggingParam = False

zoomInFactor = 2.0
zoomOutFactor = 1 / zoomInFactor

z = glm.dvec2(StartX, StartY)
znew = glm.dvec2()

oldX, oldY = float(), float()
oldX128, oldY128 = [0.0, 0.0], [0.0, 0.0]

oldStartX, oldStartY = float(), float()

layer = 0.0

i = 0

formText = """
znew.x = z.x*z.x-z.y*z.y+c.x;
znew.y = 2.0*z.x*z.y+c.y;
"""

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

slopes = False

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

#extension GL_ARB_shading_language_420pack : enable

#pragma optionNV(fastmath off)
#pragma optionNV(fastprecision off)

#define M_PI 3.14159265358

uniform int itr;
uniform double zoom;
uniform dvec2 screenSize;
uniform dvec2 offset;
uniform dvec4 offset128;

uniform bool useQuadPrec;

uniform sampler1D gradient;

uniform float layer;
uniform float time;

uniform double bailout;

double n = 0.0;
double threshold = bailout;

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
uniform vec3 col11;

uniform vec3 inColor;

in vec2 aTexcoord;
uniform dvec2 cFormula;

bool escaped = false;

vec3 palette[11] = vec3[11](vec3(col1), vec3(col2), vec3(col3), vec3(col4), vec3(col5), vec3(col6), vec3(col7), vec3(col8), vec3(col9), vec3(col10), vec3(col11));

/* 
0 - Mandelbrot
1 - Burning Ship
2 - Tricorn / Mandelbar
*/
dvec2 z;

uniform int FRACTAL_TYPE;
uniform float POWER;

uniform bool juliaEnabled;
uniform bool smoothColoring;
uniform bool perturbationEnabled;
uniform bool slopesEnabled;

uniform dvec2 Start;

#define twopi 6.283185
#define halfpi twopi / 4

// Experimental double functions

double pow_d(double a, double b) {
    int e = int(b);
    struct U {
        double d;
        int x[2];
    };
    U u;
    u.d = a;
    u.x[1] = int((b-e)*(u.x[1] - 1072632447) + 1072632447);
    u.x[0] = 0;

    double r = 1.0;
    while (bool(e)) {
        if (bool(e & 1)) {
            r *= a;
        }
        a *= a;
        e >>= 1;
    }
    return r * u.d;
}

double atan2_d(double y, double x) {
    const double ONEQTR_PI = M_PI / 4.0;
    const double THRQTR_PI = 3.0 * M_PI / 4.0;
    double r, angle;
    double abs_y = abs(y) + 1e-10;
    if (x < 0.0) {
        r = (x + abs_y) / (abs_y - x);
        angle = THRQTR_PI;
    } else {
        r = (x - abs_y) / (x + abs_y);
        angle = ONEQTR_PI;
    }
    angle += (0.1963*r*r-0.9817)*r;
    if (y < 0.0)
        return -angle;
    else
        return angle;
}

double f2 = 2.0;
double f3 = 6.0;
double f4 = 24.0;
double f5 = 120.0;
double f6 = 720.0;
double f7 = 5040.0;
double f8 = 40320.0;
double f9 = f8*9.0;
double f10 = f9*10.0;
double f11 = f10*11.0;
double f12 = f11*12.0;
double f13 = f12*13.0;
double f14 = f13*14.0;
double f15 = f14*15.0;

double fact(double x) {
    double res = 1.0;
    for (double i = 1.0; i <= x; i++) {
        res *= i;
    }
    return res;
}

double sin_d(double x) {
    int i = 1;
    double cur = x;
    double acc = 1.0;
    double fc = 1.0;
    double p = x;
    while (abs(acc) > .00000001 && i < 100) {
        fc *= ((2.0*double(i))*(2.0*double(i)+1.0));
        p *= -1.0 * (x*x);
        acc = p / fc;
        cur += acc;
        i++;
    }
    return cur;
}

double cos_d(double x) {
    double t, s;
    int p = 0;
    s = 1.0;
    t = 1.0;
    while (abs(t/s) > .00000001) {
        p++;
        t = (-t*x*x)/((2.0*double(p)-1.0)*(2.0*double(p)));
        s += t;
    }
    return s;
}

dvec2 set(double a) {
    dvec2 z;
    z.x = a;
    z.y = 0.0;
    return z;
}

dvec2 add(dvec2 dsa, dvec2 dsb) {
    dvec2 dsc;
    double t1, t2, e;

    t1 = dsa.x + dsb.x;
    e = t1 - dsa.x;
    t2 = ((dsb.x - e) + (dsa.x - (t1 - e))) + dsa.y + dsb.y;

    dsc.x = t1 + t2;
    dsc.y = t2 - (dsc.x - t1);
    return dsc;
}

void split(precise double a, precise double hi, precise double lo) {
    precise double temp;
    if (a > 6.69692879491417e+299 || a < -6.69692879491417e+299) {
        a *= 3.7252902984619140625e-09;
        temp = 8192.0 * a;
        hi = temp - (temp - a);
        lo = a - hi;
        hi *= 268435456.0;
        lo *= 268435456.0;
    } else {
        temp = 8192.0 * a;
        hi = temp - (temp - a);
        lo = a - hi;
    }
}

double fms(precise double a, precise double b, precise double c) {
    return fma(a, b, -c);
}

double two_prod(precise double a, precise double b, precise double err) {
    double a_hi, a_lo, b_hi, b_lo;
    double p = a * b;
    split(a, a_hi, a_lo);
    split(b, b_hi, b_lo);
    err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    return p;
}

double quick_two_sum(precise double a, precise double b, precise double err) {
    precise double s = a + b;
    err = b - (s - a);
    return s;
}

dvec2 mul(dvec2 dsa, dvec2 dsb) {
    precise dvec2 dsc;
    double c11, c21, c2, e, t1, t2;
    double a1, a2, b1, b2, cona, conb;
    const double split = 134217729.0;

    cona = dsa.x * split;
    conb = dsb.x * split;
    a1 = cona - (cona - dsa.x);
    b1 = conb - (conb - dsb.x);
    a2 = dsa.x - a1;
    b2 = dsb.x - b1;

    c11 = dsa.x * dsb.x;
    c21 = a2 * b2 + (a2 * b1 + (a1 * b2 + (a1 * b1 - c11)));

    c2 = dsa.x * dsb.y + dsa.y * dsb.x;

    t1 = c11 + c2;
    e = t1 - c11;
    t2 = dsa.y * dsb.y + ((c2 - e) + (c11 - (t1 - e))) + c21;

    dsc.x = t1 + t2;
    dsc.y = t2 - (dsc.x - t1);
    return dsc;
}

/*dvec2 mul(precise dvec2 a, precise dvec2 b) {
    precise double p1, p2;
    p1 = two_prod(a.x, b.x, p2);
    p2 += (a.x * b.y + a.y * b.x);
    p1 = quick_two_sum(p1, p2, p2);
    return dvec2(p1, p2);
}*/

dvec2 sub(dvec2 dsa, dvec2 dsb) {
    return add(dsa, mul(set(-1.0), dsb));
}

double cmp(dvec2 a, dvec2 b) {
    if (a.x == b.x && a.y == b.y)
        return 0.0;
    else if (a.x > b.x || (a.x == b.x && a.y > b.y))
        return 1.0;
    else if (a.x < b.x || (a.x == b.x && a.y < b.y))
        return -1.0;
}

dvec2 abs128(dvec2 a) {
    dvec2 c;
    if (cmp(a, set(0.0)) == -1.0) {
        c.x = -a.x;
        c.y = -a.y;
    } else if (cmp(a, set(0.0)) == 0.0 || cmp(a, set(0.0)) == 1.0) {
        c = a;
    }
    return c;
}

// DOUBLE128

// DOUBLE256 (Farthest I could go)

dvec4 set256(double a) {
    precise dvec4 z;
    z.x = a;
    z.y = 0.0;
    z.z = 0.0;
    z.w = 0.0;
    return z;
}

dvec4 add256(dvec4 dsa, dvec4 dsb) {
    dvec4 dsc;
    dvec2 t1, t2, e;

    t1 = add(dsa.xy, dsb.xy);
    e = sub(t1, dsa.xy);
    t2 = add(add(add(sub(dsb.xy, e), sub(dsa.xy, sub(t1, e))), dsa.zw), dsb.zw);

    dsc.xy = add(t1, t2);
    dsc.zw = sub(t2, sub(dsc.xy, t1));
    return dsc;
}

dvec2 c_pow(dvec2 a, float power) {
    double r = sqrt(a.x*a.x+a.y*a.y);
    float theta = atan(float(a.y), float(a.x));
    dvec2 z = pow(float(r), power) * dvec2(cos(power*theta), sin(power*theta));
    return z;
}

dvec2 c_abspow(dvec2 a, float power) {
    double r = sqrt(a.x*a.x+a.y*a.y);
    float theta = atan(float(z.y), float(z.x));
    dvec2 z = pow(float(r), power) * dvec2(abs(cos(power*theta)), abs(sin(power*theta)));
    return z;
}

dvec2 c_powre(dvec2 a, float power) {
    double r = sqrt(a.x*a.x+a.y*a.y);
    float theta = atan(float(z.y), float(-abs(z.x)));
    dvec2 z = pow(float(r), power) * dvec2(cos(power*theta), sin(power*theta));
    return z;
}

dvec2 c_powim(dvec2 a, float power) {
    double r = sqrt(a.x*a.x+a.y*a.y);
    float theta = atan(float(-abs(z.y)), float(z.x));
    dvec2 z = pow(float(r), power) * dvec2(cos(power*theta), sin(power*theta));
    return z;
}

dvec2 c_powcel(dvec2 a, float power) {
    double r = sqrt(a.x*a.x+a.y*a.y);
    float theta = atan(float(z.y), float(z.x));
    dvec2 z = pow(float(r), power) * dvec2(abs(cos(power*theta)), sin(power*theta));
    return z;
}

dvec2 c_times(dvec2 a, dvec2 b) {
    return dvec2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);
}

dvec4 c_times128(dvec4 a, dvec4 b) {
    dvec4 o;
    o.xy = sub(mul(a.xy,b.xy), mul(a.zw,b.zw));
    o.zw = add(mul(a.xy,b.zw), mul(a.xy,b.zw));
    return o;
}

dvec2 c_celtimes(dvec2 a, dvec2 b) {
    return dvec2(abs(a.x*b.x-a.y*b.y),a.x*b.y+a.y*b.x);
}

dvec2 c_celtimesre(dvec2 a, dvec2 b) {
    return dvec2(abs(a.x*b.x-a.y*b.y),-(abs(a.x)*b.y+a.y*abs(b.x)));
}

dvec2 c_celtimesim(dvec2 a, dvec2 b) {
    return dvec2(abs(a.x*b.x-a.y*b.y),-(a.x*abs(b.y)+abs(a.y)*b.x));
}

double diffabs(double c, double d) {
    double cd = c+d;
    if (c >= 0.0) {
        if (cd >= 0.0) {
            return d;
        } else {
            return -d - 2.0 * c;
        }
    } else {
        if (cd > 0.0) {
            return d + 2.0 * c;
        } else {
            return -d;
        }
    }
}

dvec2 c_powi(dvec2 a, int power) {
    dvec2 o = a;
    for (int i = 1; i < power; i++) {
        o = c_times(o, a);
    }
    return o;
}

dvec2 c_powceli(dvec2 a, int power) {
    dvec2 o = a;
    for (int i = 1; i < power; i++) {
        o = c_celtimes(o, a);
    }
    return o;
}

dvec2 c_powcelrei(dvec2 a, int power) {
    dvec2 o = a;
    for (int i = 1; i < power; i++) {
        o = c_celtimesre(o, a);
    }
    return o;
}

dvec2 c_powcelimi(dvec2 a, int power) {
    dvec2 o = a;
    for (int i = 1; i < power; i++) {
        o = c_celtimesim(o, a);
    }
    return o;
}

dvec2 c_div(dvec2 a, dvec2 b) {
    return dvec2((a.x*b.x+a.y*b.y)/(b.x*b.x+b.y*b.y), (a.y*b.x-a.x*b.y)/(b.x*b.x+b.y*b.y));
}

bool slopes = true;
bool inverted = false;

bool distanceEstimation = slopesEnabled;

double mandelbrot(dvec2 c) {
    z = vec2(Start.x, Start.y);
    dvec2 dz = dvec2(1.0, 0.0);

    if (smoothColoring) {
        double c2 = dot(c, c);
    }

    if (slopes) {
        double h2 = 1.5;
        int angle = 45;

        int R = 100;
    }

    const double B = 256.0;
    double l = 0.0;

    for (int i = 0; i < itr; i++) {
        dvec2 dznew;
        dvec2 znew;

        double zrsqr = z.x*z.x;
        double zisqr = z.y*z.y;
        double zrzrsqr = zrsqr*zrsqr;
        double zizisqr = zisqr*zisqr;
        double zrzisqr = zrsqr*zisqr;

        //dvec2 zc = c_pow(z.x, z.y, POWER);
        //dvec2 zabsr = c_pow(-abs(z.x), z.y, POWER);
        //dvec2 zabsi = c_pow(z.x, -abs(z.y), POWER);
    
        if (FRACTAL_TYPE == 0) {
            if (POWER == int(POWER)) {
                if (POWER == 2) {
                    znew = c_powi(z, int(POWER)) + c;
                    if (POWER == 2)
                        dznew = c_times(dvec2(2.0, 0.0), c_times(z, dz))+1.0;
                } else {
                    znew = c_powi(z, int(POWER)) + c;
                }
            } else
                znew = c_pow(z, POWER) + c;
        } else if (FRACTAL_TYPE == 1) {
            if (POWER == int(POWER)) {
                znew = dvec2(c_powi(z, int(POWER)).x, -(c_powi(z, int(POWER))).y) + c;
                if (POWER == 2) {
                    dznew = c_times(dvec2(2.0, 0.0), c_times(dvec2(z.x,-z.y), dvec2(dz.x,-dz.y)))+1.0;
                }
            } else
                znew = dvec2(c_pow(z, POWER).x, -c_pow(z, POWER).y) + c;
        } else if (FRACTAL_TYPE == 2) {
            if (POWER == int(POWER))
                znew = c_powi(dvec2(abs(z.x), -abs(z.y)), int(POWER)) + c;
            else {
                if (z.x < 0.0) {
                    z.x = -z.x;
                }
                if (z.y < 0.0) {
                    z.y = -z.y;
                }
                znew = dvec2(c_pow(abs(z), POWER).x, -c_pow(abs(z), POWER).y) + c;
            }
        } else if (FRACTAL_TYPE == 3) {
            if (POWER == int(POWER))
                znew = c_pow(dvec2(abs(z.x), -z.y), int(POWER)) + c;
            else
                znew = c_pow(dvec2(abs(z.x), -z.y), POWER) + c;
        } else if (FRACTAL_TYPE == 4) {
            if (POWER == int(POWER))
                znew = dvec2(c_powi(z, int(POWER)).x, -(c_powi(dvec2(z.x, abs(z.y)), int(POWER))).y) + c;
            else
                znew = c_powim(z, POWER) + c;
        } else if (FRACTAL_TYPE == 5) {
            if (POWER == int(POWER))
                znew = dvec2(abs(c_powi(z, int(POWER)).x), c_powi(z, int(POWER)).y) + c;
            else
                znew = c_powcel(z, POWER) + c;
        } else if (FRACTAL_TYPE == 6) {
            if (POWER == int(POWER))
                znew = dvec2(abs(c_powi(z, int(POWER)).x), -(c_powi(z, int(POWER)).y)) + c;
            else
                znew = dvec2(c_powcel(z, POWER).x, -c_powcel(z, POWER).y) + c;
        } else if (FRACTAL_TYPE == 7) {
            if (POWER == int(POWER))
                znew = dvec2(abs(c_powi(z, int(POWER))).x, -(c_powi(dvec2(abs(z.x), z.y), int(POWER))).y) + c;
            else
                znew = dvec2(c_powcel(z, POWER).x, -(c_pow(dvec2(abs(z.x), z.y), POWER)).y) + c;
        } else if (FRACTAL_TYPE == 8) {
            if (POWER == int(POWER))
                znew = dvec2(abs(c_powi(z, int(POWER)).x), -abs(c_powi(z, int(POWER)).y)) + c;
            else
                znew = dvec2(abs(c_pow(z, POWER)).x, -abs(c_pow(z, POWER)).y) + c;
        } else if (FRACTAL_TYPE == 9) {
            znew = c_powcelimi(z, int(POWER)) + c;
        } else if (FRACTAL_TYPE == 10) {
            if (POWER == int(POWER)) {
                if (mod(i,4)==0) {
                    znew = c_powi(z, int(POWER)) + c;
                }
                if (mod(i,4)==1) {
                    znew = dvec2(c_powi(z, int(POWER)).x, -abs(c_powi(z, int(POWER))).y) + c;
                }
                if (mod(i,4)==2) {
                    znew = c_powi(z, int(POWER)) + c;
                }
                if (mod(i,4)==3) {
                    znew = c_powi(z, int(POWER)) + c;
                }
            } else {
                if (mod(i,4)==0) {
                    znew = c_pow(z, POWER) + c;
                }
                if (mod(i,4)==1) {
                    znew = dvec2(c_pow(z, POWER).x, -abs(c_pow(z, POWER)).y) + c;
                }
                if (mod(i,4)==2) {
                    znew = c_pow(z, POWER) + c;
                }
                if (mod(i,4)==3) {
                    znew = c_pow(z, POWER) + c;
                }
            }
        } else if (FRACTAL_TYPE == 11) {
           if (POWER == int(POWER))
                znew = c_powi(dvec2(abs(z.x), z.y), int(POWER)) + c;
            else
                znew = c_pow(dvec2(abs(z.x), z.y), POWER) + c;
        } else if (FRACTAL_TYPE == 12) {
            //znew = c_powcel(abs(z.x), z.y, POWER) + c;
            if (POWER == int(POWER))
                znew = dvec2(abs(c_powi(dvec2(abs(z.x), z.y), int(POWER)).x), c_powi(dvec2(abs(z.x), z.y), int(POWER)).y) + c;
            else
                znew = dvec2(abs(c_pow(dvec2(abs(z.x), z.y), POWER).x), c_pow(dvec2(abs(z.x), z.y), POWER).y) + c;
        } else if (FRACTAL_TYPE == 13) {
            znew = c_times(c, c_times(z, dvec2(1.0, 0.0)-z));
        } else if (FRACTAL_TYPE == 14) {
"""+formText+"""
        }

        z = znew;
        if (distanceEstimation)
            dz = dznew;
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

    if (distanceEstimation) {
        double h2 = 0.75;
        double angle = 45.0;
        vec2 v = exp(vec2(c_div(c_times(dvec2(0.0, 1.0), c_times(dvec2(angle, 0.0), c_times(dvec2(2.0, 0.0), dvec2(3.14159265358, 0.0)))), dvec2(360.0, 0.0))));
        dvec2 u = c_div(z, dz);
        u = c_div(u, dvec2(dot(u,u), 0.0));
        double t = u.x*v.x+u.y*v.y;
        t = t/(1.0+h2);
        if (t < 0.0) t = 0.0;
        n = t/(zoom/8.0);
    }

    return n / float(itr);
}

double julia(dvec2 z) {
    dvec2 c = dvec2(Start.x, Start.y);

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

        //dvec2 zc = c_pow(z.x, z.y, POWER);
        //dvec2 zabsr = c_pow(-abs(z.x), z.y, POWER);
        //dvec2 zabsi = c_pow(z.x, -abs(z.y), POWER);
    
        if (FRACTAL_TYPE == 0) {
            if (POWER == int(POWER))
                znew = c_powi(z, int(POWER)) + c;
            else
                znew = c_pow(z, POWER) + c;
        } else if (FRACTAL_TYPE == 1) {
            if (POWER == int(POWER))
                znew = dvec2(c_powi(z, int(POWER)).x, -(c_powi(z, int(POWER))).y) + c;
            else
                znew = dvec2(c_pow(z, POWER).x, -c_pow(z, POWER).y) + c;
        } else if (FRACTAL_TYPE == 2) {
            if (POWER == int(POWER))
                znew = dvec2(c_powi(z, int(POWER)).x, -abs(c_powi(z, int(POWER))).y) + c;
            else
                znew = dvec2(c_pow(z, POWER).x, -abs(c_pow(z, POWER)).y) + c;
        } else if (FRACTAL_TYPE == 3) {
            if (POWER == int(POWER))
                znew = c_powi(dvec2(-abs(z.x), z.y), int(POWER)) + c;
            else
                znew = c_powre(z, POWER) + c;
        } else if (FRACTAL_TYPE == 4) {
            if (POWER == int(POWER))
                znew = dvec2(c_powi(z, int(POWER)).x, -(c_powi(dvec2(z.x, abs(z.y)), int(POWER))).y) + c;
            else
                znew = c_powim(z, POWER) + c;
        } else if (FRACTAL_TYPE == 5) {
            if (POWER == int(POWER))
                znew = dvec2(abs(c_powi(z, int(POWER)).x), c_powi(z, int(POWER)).y) + c;
            else
                znew = c_powcel(z, POWER) + c;
        } else if (FRACTAL_TYPE == 6) {
            if (POWER == int(POWER))
                znew = dvec2(abs(c_powi(z, int(POWER)).x), -(c_powi(z, int(POWER)).y)) + c;
            else
                znew = c_powcel(dvec2(z.x, -z.y), POWER) + c;
        } else if (FRACTAL_TYPE == 7) {
            if (POWER == int(POWER))
                znew = dvec2(abs(c_powi(z, int(POWER))).x, -(c_powi(dvec2(abs(z.x), z.y), int(POWER))).y) + c;
            else
                znew = dvec2(c_powcel(z, POWER).x, -(c_pow(dvec2(abs(z.x), z.y), POWER)).y) + c;
        } else if (FRACTAL_TYPE == 8) {
            if (POWER == int(POWER))
                znew = dvec2(c_powcel(z, int(POWER)).x, -c_powcelrei(z, int(POWER)).y) + c;
            else
                znew = dvec2(abs(c_pow(z, POWER)).x, -abs(c_pow(z, POWER)).y) + c;
        } else if (FRACTAL_TYPE == 9) {
            znew = c_powcelimi(z, int(POWER)) + c;
        } else if (FRACTAL_TYPE == 10) {
            if (POWER == int(POWER)) {
                if (mod(i,4)==0) {
                    znew = c_powi(z, int(POWER)) + c;
                }
                if (mod(i,4)==1) {
                    znew = dvec2(c_powi(z, int(POWER)).x, -abs(c_powi(z, int(POWER))).y) + c;
                }
                if (mod(i,4)==2) {
                    znew = c_powi(z, int(POWER)) + c;
                }
                if (mod(i,4)==3) {
                    znew = c_powi(z, int(POWER)) + c;
                }
            } else {
                if (mod(i,4)==0) {
                    znew = c_pow(z, POWER) + c;
                }
                if (mod(i,4)==1) {
                    znew = dvec2(c_pow(z, POWER).x, -abs(c_pow(z, POWER)).y) + c;
                }
                if (mod(i,4)==2) {
                    znew = c_pow(z, POWER) + c;
                }
                if (mod(i,4)==3) {
                    znew = c_pow(z, POWER) + c;
                }
            }
        } else if (FRACTAL_TYPE == 11) {
           if (POWER == int(POWER))
                znew = c_powi(dvec2(abs(z.x), z.y), int(POWER)) + c;
            else
                znew = c_pow(dvec2(abs(z.x), z.y), POWER) + c;
        } else if (FRACTAL_TYPE == 12) {
            //znew = c_powcel(abs(z.x), z.y, POWER) + c;
            if (POWER == int(POWER))
                znew = dvec2(abs(c_powi(dvec2(abs(z.x), z.y), int(POWER)).x), c_powi(dvec2(abs(z.x), z.y), int(POWER)).y) + c;
            else
                znew = dvec2(abs(c_pow(dvec2(abs(z.x), z.y), POWER).x), c_pow(dvec2(abs(z.x), z.y), POWER).y) + c;
        } /*else if (FRACTAL_TYPE == 13) {
            znew.x = z.x*z.x - z.y*z.y + c.x;
            znew.y = 2.0 * z.x  * z.y + c.y;
        }*/

        z = znew;
        if(dot(z, z) > threshold) {
            escaped = true;
            break;
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

double mandelInv(dvec2 c) {
    return mandelbrot(c_div(dvec2(1.0, 0.0), c));
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

vec3 mapToColor(float t, vec3 c1, vec3 c2, vec3 c3, vec3 c4, vec3 c5, vec3 c6, vec3 c7, vec3 c8, vec3 c9, vec3 c10, vec3 c11) {
    if (escaped) {
	    if (t < x) return mix(c1, c2, t/x);
	    else if (t < 2.0 * x) return mix(c2, c3, (t - x)/x);
	    else if (t < 3.0 * x) return mix(c3, c4, (t - 2.0*x)/x);
	    else if (t < 4.0 * x) return mix(c4, c5, (t - 3.0*x)/x);
	    else if (t < 5.0 * x) return mix(c5, c6, (t - 4.0*x)/x);
	    else if (t < 6.0 * x) return mix(c6, c7, (t - 5.0*x)/x);
	    else if (t < 7.0 * x) return mix(c7, c8, (t - 6.0*x)/x);
	    else if (t < 8.0 * x) return mix(c8, c9, (t - 7.0*x)/x);
	    else if (t < 9.0 * x) return mix(c9, c10, (t - 8.0*x)/x);
	    else if (t < 10.0 * x) return mix(c10, c11, (t - 9.0*x)/x);

	    return c10;
    } else
        return inColor;
}

bool glitchedPixel = false; 
double mandelbrotPerturbation(dvec2 c, dvec2 dc) {
    dvec2 z = dvec2(Start.x, Start.y);
    dvec2 dz = dvec2(Start.x, Start.y);
    dvec2 glitchedC = dvec2(0.0, 0.0);
    for (int i = 0; i < itr; i++) {
        dvec2 dznew;
        dvec2 znew;
        if (FRACTAL_TYPE == 0) {
            dz = c_times(2.0*z+dz,dz) + dc;
            //dz = c_times(dvec2(2.0, 0.0), c_times(z, dn)) + c_powi(dn, 2) + dc;
            z = c_times(z, z) + c;
            if (glitchedPixel) {
                dz = c_times(2.0*z+dz, dz) + glitchedC;
                z = c_times(z, z) + c;
            }
        } else if (FRACTAL_TYPE == 1) {
            dznew.x = 2.0*z.x*dz.x+(dz.x*dz.x)-(dz.y*dz.y)-2.0*dz.y*z.y + dc.x;
            dznew.y = dc.y - 2.0*(z.x*dz.y+dz.x*z.y+dz.x*dz.y);
            znew.x = z.x*z.x - z.y*z.y + c.x;
            znew.y = c.y - (2.0*z.x*z.y);
            dz = dznew;
            z = znew;
        } else if (FRACTAL_TYPE == 2) {
            dznew.x = 2.0*z.x*dz.x+(dz.x*dz.x)-(dz.y*dz.y)-2.0*dz.y*z.y + dc.x;
            dznew.y = dc.y - 2.0*diffabs(z.x*z.y,z.x*dz.y+dz.x*z.y+dz.x*dz.y);
            znew.x = z.x*z.x - z.y*z.y + c.x;
            znew.y = -2.0*abs(z.x*z.y)+c.y;
            dz = dznew;
            z = znew;
        }
        //dz = dznew;
        //z = znew;
        if (dot(z+dz, z+dz) > threshold) {
            escaped = true;
            break;
        }
        if (dot(z+dz, z+dz)/dot(z, z) < 1e-6) {
            glitchedPixel = true;
            glitchedC = z + dz;
            break;
        }
        n++;
    }
    return n/float(itr);
}

dvec2[100] deepZoomPoint(int depth) {
    dvec2[100] v;
    double xn_r = offset.x;
    double xn_i = offset.y;

    for (int i = 0; i != depth; i++) {
        double re = xn_r + xn_r;
        double im = xn_i + xn_i;

        dvec2 c = dvec2(re, im);
        v[i] = c;

        if (re > 1024.0 || im > 1024.0 || re < -1024.0 || im < -1024.0)
            return v;
        
        xn_r = xn_r * xn_r - xn_i * xn_i + offset.x;
        xn_i = re * xn_i + offset.y;
    }
    return v;
}

double mandelbrot128(dvec2 cx, dvec2 cy) {
    dvec4 z = dvec4(0.0);
    double c2 = dot(cx.x, cy.x);
    const double B = 256.0;
    double l = 0.0;
    for (int i = 0; i < itr; i++) {
        dvec4 znew;
        if (FRACTAL_TYPE == 0) {
            znew.xy = add(sub(mul(z.xy, z.xy), mul(z.zw, z.zw)), cx);
            znew.zw = add(add(mul(z.xy, z.zw), mul(z.xy, z.zw)), cy);
        } else if (FRACTAL_TYPE == 1) {
            znew.xy = add(sub(mul(z.xy, z.xy), mul(z.zw, z.zw)), cx);
            znew.zw = add(mul(set(-2.0), mul(z.xy,z.zw)), cy);
        } else if (FRACTAL_TYPE == 2) {
            znew.xy = add(sub(mul(z.xy, z.xy), mul(z.zw, z.zw)), cx);
            znew.zw = add(mul(set(-2.0), abs128(mul(z.xy,z.zw))), cy);
        } else if (FRACTAL_TYPE == 3) {
            znew.xy = add(sub(mul(z.xy, z.xy), mul(z.zw, z.zw)), cx);
            znew.zw = add(mul(set(-2.0), mul(abs128(z.xy), z.zw)), cy);
        } else if (FRACTAL_TYPE == 4) {
            znew.xy = add(sub(mul(z.xy, z.xy), mul(z.zw, z.zw)), cx);
            znew.zw = add(mul(set(-2.0), mul(z.xy, abs128(z.zw))), cy);
        } else if (FRACTAL_TYPE == 5) {
            znew.xy = add(abs128(sub(mul(z.xy, z.xy), mul(z.zw, z.zw))), cx);
            znew.zw = add(add(mul(z.xy, z.zw), mul(z.xy, z.zw)), cy);
        } else if (FRACTAL_TYPE == 6) {
            znew.xy = add(abs128(sub(mul(z.xy, z.xy), mul(z.zw, z.zw))), cx);
            znew.zw = add(mul(set(-2.0), mul(z.xy, z.zw)), cy);
        }
        z = znew;
        if (cmp(add(mul(z.xy, z.xy), mul(z.zw, z.zw)), set(threshold)) == 1.0) {
            escaped = true;
            break;
        }
        n++;
    }

    double sl = n - log2(log2(float(dot(z.xz,z.xz)))) + 4.0;

    double al = smoothstep(-0.1, 0.0, sin(0.5*6.2831));
    n = mix(n, sl, al);

    return n/float(itr);
}

double mandelbrotPerturbation128(dvec2 cx, dvec2 cy, dvec2 dcx, dvec2 dcy) {
    dvec4 z = dvec4(set(Start.x), set(Start.y));
    dvec4 dz = dvec4(set(Start.x), set(Start.y));
    dvec4 glitchedC = dvec4(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < itr; i++) {
        dvec4 dznew;
        dvec4 znew;
        dznew = dvec4(add(c_times128(dvec4(add(mul(set(2.0),z.xy),dz.xy), add(mul(set(2.0),z.zw),dz.zw)), dz).xy, dcx), add(c_times128(dvec4(add(mul(set(2.0),z.xy),dz.xy), add(mul(set(2.0),z.zw),dz.zw)), dz).zw, dcy));
        znew.xy = add(sub(mul(z.xy, z.xy), mul(z.zw, z.zw)), cx);
        znew.zw = add(add(mul(z.xy, z.zw), mul(z.xy, z.zw)), cy);
        dz = dznew;
        z = znew;
        if (cmp(mul(mul(add(z.xy,dz.xy), add(z.xy,dz.xy)),mul(add(z.zw,dz.zw), add(z.zw,dz.zw))), set(threshold)) == 1.0) {
            escaped = true;
            break;
        }
        n++;
    }
    return n/float(itr);
}

int glitchedPixels = 0;

void main() {
    const dvec2 coord = dvec2(gl_FragCoord.xy);
    vec2 resolution = vec2(3.0, 3.0);
    vec2 direction = vec2(1.0, 0.5);
    vec2 off1 = vec2(1.3846153846) * direction;
    vec2 off2 = vec2(3.2307692308) * direction;
    dvec2 z = vec2(Start.x, Start.y);
    dvec2 p = (coord - screenSize / 2.0);
    dvec2 p2 = (coord - screenSize / 2.0) / zoom - offset;
    dvec2 znew;

    double t;
    if (!juliaEnabled) {
        dvec2 c = -offset;
        dvec2 dc = p / zoom;
        
        if (inverted) {
            t = mandelInv(c + dc);
        } else {
            //double t2 = mandelbrot(c + dc);
            if (useQuadPrec) {
                if (zoom >= 10e+15)
                    t = mandelbrot128(sub(mul(sub(set(coord.x), mul(set(screenSize.x), set(0.5))), set(1.0/zoom)), offset128.xy), sub(mul(sub(set(coord.y), mul(set(screenSize.y), set(0.5))), set(1.0/zoom)), offset128.zw));
                else
                    t = perturbationEnabled ? mandelbrotPerturbation(c, dc) : mandelbrot(c + dc);
            } else
                t = perturbationEnabled ? mandelbrotPerturbation(c, dc) : mandelbrot(c + dc);
        }
        dc += -offset - c;
    } else
        t = julia(((coord - screenSize / 2) / zoom) - offset);

    //double t3d;
    //t3d = mandelbrot3d(((coord - screenSize / 2) / zoom) - offset, 1.0);

    if (gl_FragCoord.x < 40)
        gl_FragColor = vec4(1.0);

    vec4 color = escaped && !glitchedPixel ? vec4(mapToColor(float(mod(t, itr)), vec3(palette[0]), vec3(palette[1]), vec3(palette[2]), vec3(palette[3]), vec3(palette[4]), vec3(palette[5]), vec3(palette[6]), vec3(palette[7]), vec3(palette[8]), vec3(palette[9]), vec3(palette[10])), 1.0) : vec4(inColor, 1.0);
    //color += escaped && !glitchedPixel ? vec4(mapToColor(float(mod(t, itr)), vec3(palette[0]), vec3(palette[1]), vec3(palette[2]), vec3(palette[3]), vec3(palette[4]), vec3(palette[5]), vec3(palette[6]), vec3(palette[7]), vec3(palette[8]), vec3(palette[9]), vec3(palette[10])), 1.0) : vec4(inColor, 1.0);

    //color.rgb += (t < 0.0) ? vec3(0.0) : 0.5 + 0.5*cos(vec3(pow(float(zoom), 0.22)*t*0.05) + vec3(3.0, 3.5, 4.0));

    //color.rgb += (t < 0.0) ? vec3(0.0) : 0.5 + 0.5*cos(vec3(pow(float(zoom), 0.22)*t*0.05) + vec3(3.0, 3.5, 4.0));

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

#define PI 3.14159265359
#define rot(a) mat2(cos(a + PI*0.5*vec4(0,1,3,0)))

float hash13(vec3 p3) {
    p3 = fract(p3 * .1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x+p3.y) * p3.z);
}

void main() {
    vec4 color = texture2D(tex, TexCoords.st);
    vec4 color2 = texture2D(tex, TexCoords.st - 0.01);
    
    color2 = color;

    gl_FragColor = vec4(1.0f-color2.rgb, 0.5f);
}
"""

program = None
scrProgram = None

numFrames = 0
lastTime = 0.0

postProcessing = True

needsClearing = False

def countFPS():
    global numFrames
    global lastTime
    global needsClearing
    curTime = glfw.get_time()
    numFrames += 1
    if (curTime - lastTime >= 1.0):
        print(min(numFrames, 60.0), "FPS")
        numFrames = 0
        lastTime += 1.0
        needsClearing = True

def set(a):
    return [a, 0.0]

def add(dsa, dsb):
    dsc = [0.0, 0.0]
    e = 0.0

    t1 = dsa[0] + dsb[0]
    e = t1 - dsa[0]
    t2 = ((dsb[0] - e) + (dsa[0] - (t1 - e))) + dsa[1] + dsb[1]

    dsc[0] = t1 + t2
    dsc[1] = t2 - (dsc[0] - t1)
    return dsc

def mul(dsa, dsb):
    split = 8193

    cona = dsa[0] * split
    conb = dsb[0] * split
    a1 = cona - (cona - dsa[0])
    b1 = conb - (conb - dsb[0])
    a2 = dsa[0] - a1
    b2 = dsb[0] - b1

    c11 = dsa[0] * dsb[0]
    c21 = a2 * b2 + (a2 * b1 + (a1 * b2 + (a1 * b1 - c11)))

    c2 = dsa[0] * dsb[1] + dsa[1] * dsb[0]

    t1 = c11 + c2
    e = t1 - c11
    t2 = dsa[1] * dsb[1] + ((c2 - e) + (c11 - (t1 - e))) + c21

    dsc = [0.0, 0.0]
    dsc[0] = t1 + t2
    dsc[1] = t2 - (dsc[0] - t1)
    return dsc

def sub(dsa, dsb):
    return add(dsa, mul(set(-1.0), dsb))

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

class RenderThread(QThread):
    def __init__(self, OpenGLWidget):
        QThread.__init__(self)
        self.d = OpenGLWidget

    def setViewportSize(self, width, height):
        self.d.mutex.lock()
        self.d.viewportWidth = width
        self.d.viewportHeight = height
        self.d.mutex.unlock()

    def start(self):
        if (self.isRunning()):
            return
        
        self.d.render = True
        if not self.isRunning():
            self.start(QThread.NormalPriority)

    def stop(self):
        self.d.mutex.lock()
        self.d.render = False
        self.d.mutex.unlock()

        self.quit()
        self.wait()
    
    def run(self):
        global program
        while True:
            self.d.mutex.lock()
            render = self.d.render
            width = self.d.viewportWidth
            height = self.d.viewportHeight
            self.d.mutex.unlock()

            if not render:
                break

            widget = self.d
            if not widget:
                break

            widget.makeCurrent()

            if not self.d.initialized():
                self.d.initialized = True

            gl.glViewport(0, 0, width, height)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            aspect = width / float(height)

            GLU.gluPerspective(45.0, aspect, 1.0, 100.0)
            gl.glMatrixMode(gl.GL_MODELVIEW)

            self.widget.update()
            gl.glUseProgram(program)
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
            gl.glUseProgramObjectARB(program)

            gl.glUniform2d(gl.glGetUniformLocation(program, "screenSize"), decimal.Decimal(550), decimal.Decimal(420))
            gl.glUniform2d(gl.glGetUniformLocation(program, "offset"), offsetX, offsetY)
            gl.glUniform1d(gl.glGetUniformLocation(program, "zoom"), zoom)
            gl.glUniform1i(gl.glGetUniformLocation(program, "itr"), itr)
            gl.glUniform1i(gl.glGetUniformLocation(program, "FRACTAL_TYPE"), curFractal)
            gl.glUniform1f(gl.glGetUniformLocation(program, "POWER"), power)
            gl.glUniform2d(gl.glGetUniformLocation(program, "Start"), StartX, StartY)
            gl.glUniform1f(gl.glGetUniformLocation(program, "juliaEnabled"), isJulia)
            gl.glUniform1f(gl.glGetUniformLocation(program, "perturbationEnabled"), perturbationEnabled)
            # gl.glUniform1i(gl.glGetUniformLocation(program, "gradient"), 0)
            gl.glUniform1f(gl.glGetUniformLocation(program, "layer"), layer)
            gl.glUniform2d(gl.glGetUniformLocation(program, "cFormula"), z.x*z.x-z.y*z.y, 2.0*abs(z.x)*z.y)
            gl.glUniform1f(gl.glGetUniformLocation(program, "smoothColoring"), smoothColoring)
            gl.glUniform1f(gl.glGetUniformLocation(program, "slopesEnabled"), slopes)
            gl.glUniform1f(gl.glGetUniformLocation(program, "time"), glfw.get_time())

            gl.glUniform3f(gl.glGetUniformLocation(program, "col1"), col1R / 100, col1G / 100 ,col1B / 100)
            gl.glUniform3f(gl.glGetUniformLocation(program, "col2"), col2R / 100, col2G / 100, col2B / 100)
            gl.glUniform3f(gl.glGetUniformLocation(program, "col3"), col3R / 100, col3G / 100, col3B / 100)
            gl.glUniform3f(gl.glGetUniformLocation(program, "col4"), col4R / 100, col4G / 100, col4B / 100)
            gl.glUniform3f(gl.glGetUniformLocation(program, "col5"), col5R / 100, col5G / 100, col5B / 100)
            gl.glUniform3f(gl.glGetUniformLocation(program, "col6"), col6R / 100, col6G / 100, col6B / 100)
            gl.glUniform3f(gl.glGetUniformLocation(program, "col7"), col7R / 100, col7G / 100, col7B / 100)
            gl.glUniform3f(gl.glGetUniformLocation(program, "col8"), col8R / 100, col8G / 100, col8B / 100)
            gl.glUniform3f(gl.glGetUniformLocation(program, "col9"), col9R / 100, col9G / 100, col9B / 100)
            gl.glUniform3f(gl.glGetUniformLocation(program, "inColor"), inColorR / 255, inColorG / 255, inColorB / 255)

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

                gl.glUniform1f(gl.glGetUniformLocation(scrProgram, "time"), glfw.get_time())
                gl.glBindTexture(gl.GL_TEXTURE_2D, renderedTexture)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                gl.glBindVertexArray(0)
            
            gl.glDeleteShader(compileShader(vertexShader, gl.GL_VERTEX_SHADER))
            gl.glDeleteShader(compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

            gl.glDeleteProgram(program)

            self.widget.update()
            
            self.widget.swapBuffers()
            self.widget.doneCurrent()

class GLWidget(QGLWidget):
    global zoom
    global program
    global offsetX
    global offsetY
    global col1R
    global col1G
    global col1B
    global fbo

    def __init__(self, parent=None):
        self.parent = parent
        QGLWidget.__init__(self, parent)
        self.lastDragPos = QPoint()
        self.mutex = QMutex()
        self.setCursor(Qt.CrossCursor)
        self.setAutoFillBackground(False)

        self.timer = QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.displayFPS)
        self.timer.start()

        self.label = QLabel("FPS", self)
        self.label.setText("                       ")
        self.label.move(10, 10)
        self.label.setStyleSheet("""
        QLabel {
            color: #FFFFFF;
            background-color: #000000;
        }
        """)

    def displayFPS(self):
        global needsClearing
        global lastTime
        global numFrames
        nf = 0.0
        lt = 0.0
        curTime = glfw.get_time()
        numFrames += 0.1
        if (curTime - lastTime >= 0.1):
            self.label.setText("FPS: " + str(int(numFrames*100.0/2.0)))
            lastTime += 0.1
            numFrames = 0.0

    def initializeGL(self):
        global program
        self.qglClearColor(QColor(255, 0, 0))
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        glFormat = QSurfaceFormat()
        glFormat.setVersion(4, 1)
        glFormat.setProfile(QSurfaceFormat.CoreProfile)
        QSurfaceFormat.setDefaultFormat(glFormat)

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))
        scrProgram = compileProgram(compileShader(screenVert, gl.GL_VERTEX_SHADER), compileShader(screenFragment, gl.GL_FRAGMENT_SHADER))

    def resizeGL(self, width, height):
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / float(height)

        GLU.gluPerspective(45.0, aspect, 1.0, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self):
        self.qglClearColor(QColor(0,0,255,255))
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
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

        gl.glBindVertexArray(vao)

        try:
            gl.glUseProgram(program)
        except:
            pass
        #gl.glValidateProgram(program)

        gl.glUniform2d(gl.glGetUniformLocation(program, "screenSize"), 550.0, 420.0)
        gl.glUniform2d(gl.glGetUniformLocation(program, "offset"), offsetX, offsetY)
        gl.glUniform4d(gl.glGetUniformLocation(program, "offset128"), offsetX128[0], offsetX128[1], offsetY128[0], offsetY128[1])
        gl.glUniform1f(gl.glGetUniformLocation(program, "useQuadPrec"), useQuadPrec)
        gl.glUniform1d(gl.glGetUniformLocation(program, "zoom"), zoom)
        gl.glUniform1i(gl.glGetUniformLocation(program, "itr"), itr)
        gl.glUniform1i(gl.glGetUniformLocation(program, "FRACTAL_TYPE"), curFractal)
        gl.glUniform1f(gl.glGetUniformLocation(program, "POWER"), power)
        gl.glUniform2d(gl.glGetUniformLocation(program, "Start"), StartX, StartY)
        gl.glUniform1f(gl.glGetUniformLocation(program, "juliaEnabled"), isJulia)
        gl.glUniform1f(gl.glGetUniformLocation(program, "perturbationEnabled"), perturbationEnabled)
        # gl.glUniform1i(gl.glGetUniformLocation(program, "gradient"), 0)
        gl.glUniform1f(gl.glGetUniformLocation(program, "layer"), layer)
        gl.glUniform2d(gl.glGetUniformLocation(program, "cFormula"), z.x*z.x-z.y*z.y, 2.0*abs(z.x)*z.y)
        gl.glUniform1f(gl.glGetUniformLocation(program, "smoothColoring"), smoothColoring)
        gl.glUniform1f(gl.glGetUniformLocation(program, "time"), glfw.get_time())
        gl.glUniform1d(gl.glGetUniformLocation(program, "bailout"), escRad)

        gl.glUniform3f(gl.glGetUniformLocation(program, "col1"), col1R / 100, col1G / 100 ,col1B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col2"), col2R / 100, col2G / 100, col2B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col3"), col3R / 100, col3G / 100, col3B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col4"), col4R / 100, col4G / 100, col4B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col5"), col5R / 100, col5G / 100, col5B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col6"), col6R / 100, col6G / 100, col6B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col7"), col7R / 100, col7G / 100, col7B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col8"), col8R / 100, col8G / 100, col8B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col9"), col9R / 100, col9G / 100, col9B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "inColor"), inColorR / 255, inColorG / 255, inColorB / 255)

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        if (postProcessing):
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

            gl.glClearColor(1., 0., 0., 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glDisable(gl.GL_DEPTH_TEST)

            gl.glBindVertexArray(scrVao)
            #gl.glUseProgram(scrProgram)

            #gl.glUniform1f(gl.glGetUniformLocation(scrProgram, "time"), glfw.get_time())
            gl.glBindTexture(gl.GL_TEXTURE_2D, renderedTexture)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glBindVertexArray(0)
        
        gl.glDeleteShader(compileShader(vertexShader, gl.GL_VERTEX_SHADER))
        gl.glDeleteShader(compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        gl.glDeleteProgram(program)

    def wheelEvent(self, event):
        global zoom
        global program
        global offsetX
        global offsetY
        global offsetX128
        global offsetY128
        global itr

        #print(offsetX, offsetY)
        #print(offsetX128, offsetY128)

        #program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        dx = (event.pos().x() - 550 / 2) / zoom - offsetX
        dy = (420 - event.pos().y() - 420 / 2) / zoom - offsetY

        offsetX = -dx
        offsetY = -dy

        dx128 = sub(mul(sub(set(event.pos().x()), mul(set(550.0), set(0.5))), set(1.0/zoom)), offsetX128)
        dy128 = sub(mul(sub(sub(set(420.0), set(event.pos().y())), mul(set(420.0), set(0.5))), set(1.0/zoom)), offsetY128)

        offsetX128 = [-dx128[0], -dx128[1]]
        offsetY128 = [-dy128[0], -dy128[1]]

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

        dx128 = mul(sub(set(event.pos().x()), mul(set(550.0), set(0.5))), set(1.0/zoom))
        dy128 = mul(sub(sub(set(420.0), set(event.pos().y())), mul(set(420.0), set(0.5))), set(1.0/zoom))

        offsetX128 = add(offsetX128, dx128)
        offsetY128 = add(offsetY128, dy128)

        thread("Update Thread", self.update).start()

        gl.glUniform1d(gl.glGetUniformLocation(program, "zoom"), zoom)
        gl.glUniform2d(gl.glGetUniformLocation(program, "offset"), offsetX, offsetY)
        gl.glUniform4d(gl.glGetUniformLocation(program, "offset128"), offsetX128[0], offsetX128[1], offsetY128[0], offsetY128[1])
        gl.glUniform1i(gl.glGetUniformLocation(program, "itr"), itr)

    def mousePressEvent(self, event):
        global dragging
        global oldX
        global oldY
        global oldX128
        global oldY128
        global oldStartX
        global oldStartY
        global offsetX
        global offsetY
        global offsetX128
        global offsetY128
        global zoom
        global juliaTrigger
        global draggingParam

        thread("Update Thread", self.update).start()

        if event.button() == Qt.LeftButton:
            dragging = True
            draggingParam = False

            oldX = event.pos().x()
            oldY = event.pos().y()
            oldX128 = set(event.pos().x())
            oldY128 = set(event.pos().y())

        if event.button() == Qt.RightButton and juliaTrigger.isChecked():
            draggingParam = True
            dragging = False

            oldStartX = event.pos().x()
            oldStartY = event.pos().y()

    def mouseMoveEvent(self, event):
        global dragging
        global offsetX
        global offsetY
        global offsetX128
        global offsetY128
        global oldX
        global oldY
        global oldX128
        global oldY128
        global oldStartX
        global oldStartY
        global program
        global zoom
        global StartX
        global StartY
        global juliaTrigger

        #program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        thread("Update Thread", self.update).start()

        if dragging:
            offsetX += (event.pos().x() - oldX) / zoom
            offsetY += (oldY - event.pos().y()) / zoom

            oldX = event.pos().x()
            oldY = event.pos().y()

            offsetX128 = add(offsetX128, mul(sub(set(event.pos().x()), oldX128), set(1.0/zoom)))
            offsetY128 = add(offsetY128, mul(sub(oldY128, set(event.pos().y())), set(1.0/zoom)))

            oldX128 = set(event.pos().x())
            oldY128 = set(event.pos().y())

            gl.glUniform2d(gl.glGetUniformLocation(program, "offset"), float(offsetX), float(offsetY))
            gl.glUniform4d(gl.glGetUniformLocation(program, "offset128"), offsetX128[0], offsetX128[1], offsetY128[0], offsetY128[1])

        if draggingParam and juliaTrigger.isChecked():
            StartX += (event.pos().x() - oldStartX) / zoom
            StartY += (oldStartY - event.pos().y()) / zoom

            oldStartX = event.pos().x()
            oldStartY = event.pos().y()

            StartRe.setText(str(StartX))
            StartIm.setText(str(StartY))

            gl.glUniform2d(gl.glGetUniformLocation(program, "Start"), float(StartX), float(StartY))

    def mouseReleaseEvent(self, event):
        global dragging
        global offsetX
        global offsetY
        global offsetX128
        global offsetY128
        global oldX
        global oldY
        global oldX128
        global oldY128
        global oldStartX
        global oldStartY
        global StartX
        global StartY
        global juliaTrigger
        global StartRe
        global StartIm
        global draggingParam

        if event.button() == Qt.LeftButton:
            offsetX += event.pos().x() - oldX
            offsetY += oldY - event.pos().y()

            offsetX128 = add(offsetX128, sub(set(event.pos().x()), oldX128))
            offsetY128 = add(offsetY128, sub(oldY128, set(event.pos().y())))

            dragging = False
            draggingParam = False

        if event.button() == Qt.RightButton and juliaTrigger.isChecked():

            dragging = False
            draggingParam = False

        thread("Update Thread", self.update).start()

    def threadedPaintGL(self):
        pass

    def keyPressEvent(self, event:QKeyEvent):
        global zoom
        global program

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))
        if event.key() == Qt.Key_Plus:
            zoom *= 1.3
        elif event.key() == Qt.Key_Minus:
            zoom /= 1.3
        else:
            super(GLWidget, self).keyPressEvent(event)

        thread("Update Thread", self.update).start()

        gl.glUniform1d(gl.glGetUniformLocation(program, "zoom"), zoom)

    def zoomIn(self, zoomFactor):
        global zoom
        global program

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        zoom *= zoomFactor

        thread("Update Thread", self.update).start()

        gl.glUniform1d(gl.glGetUniformLocation(program, "zoom"), zoom)

class thread(threading.Thread):
    def __init__(self, thread_name, func):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.func = func
        self.thread_finished = False
    def run(self):
        print("Starting " + self.thread_name)
        self.func()
        print("Exiting " + self.thread_name)
        self.thread_finished = True

class MainWindow(QMainWindow):
    def __init__(self):
        global program
        global vertexShader
        global fragmentShader
        QMainWindow.__init__(self)

        self.resize(WIDTH + 20, 490)
        self.setWindowTitle("UniFract II")

        self.glWidget = GLWidget(self)
        self.glWidget.resize(550, 420)
        self.glWidget.move(5, 65)

        self.fractalEditor = fractalEditor(self, self.glWidget, program, vertexShader, fragmentShader)
        self.initUI()

    def initUI(self):
        global fractalType
        global program
        global power
        global colorMethod
        global CoordX
        global CoordY
        global precision
        global perturb
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
        global R5
        global G5
        global B5
        global zlay
        global formulaBtn
        global presCol
        global bailout
        global useSlopes

        # Set UI Style

        if False:
            self.setStyleSheet("""
            QPushButton, QDoubleSpinBox, QComboBox, QLineEdit {
                background-color: #00BBFF;
                color: #FFFFFF;
                border: none;
                border-radius: 2px;
            }

            QSlider::handle, QSlider::sub-page:vertical {
                background: #000011;
            }

            QComboBox::drop-down {
                background-color: #0097CF;
                color: #000000;
            }

            QComboBox::drop-down:on {
                padding-left: 4px;
            }
            
            QComboBox QAbstractItemView {
                background-color: #000022;
            }

            QPushButton:hover {
                background-color: #7ADCFF;
            }

            QPushButton:pressed {
                background-color: #0097CF;
            }

            QCheckBox::indicator {
                background-color: #00BBFF;
                color: #FFFFFF;
            }

            QMainWindow {
                background-color: #000011;
            }

            QLabel {
                color: #FFFFFF;
            }

            QTabWidget {
                background-color: #001155;
                color: #FFFFFF;
                border: none;
                border-radius: 2px;
            }
            QTabBar {
                background-color: #002277;
                color: #FFFFFF;
                border: none;
                border-radius: 2px;
            }

            QTabBar::tab:selected {
                background-color: #3064E6;
                color: #FFFFFF;
            }

            QTabBar::tab:hover {
                background-color: #002EA1;
            }

            QTabBar QToolButton {
                background-color: #00BBFF;
                color: #000011;
                border: none;
                border-radius: 2px;
            }

            QTabBar QToolButton::hover {
                background-color: #7ADCFF;
                color: #FFFFFF;
            }

            QDockWidget::close-button {
                border: none;
                background: transparent;
                icon-size: 12px;
            }
            """)

        # FPS Label
        self.fps = QLabel()
        self.fps.move(6, 66)

        # Tabs

        tabs = QTabWidget(self)
        tab1 = QWidget(self)
        tab2 = QWidget(self)
        tab3 = QWidget(self)
        tab4 = QWidget(self)
        tabs.resize(255, 420)

        tabs.move(560, self.glWidget.y())

        tabs.addTab(tab1, "Fractal")
        tabs.addTab(tab2, "Rendering")
        tabs.addTab(tab3, "Gradient")
        tabs.addTab(tab4, "Other")

        tab1.layout = QFormLayout()
        tab2.layout = QFormLayout()
        tab3.layout = QFormLayout()
        tab4.layout = QFormLayout()

        # Main Menu

        mainMenu = QMenuBar(self)
        fileMenu = mainMenu.addMenu("&File")
        editMenu = mainMenu.addMenu("&Edit")
        exportImgSeries = fileMenu.addAction("Export Image Series")
        exportImgSeries.triggered.connect(self.exportImgs)

        # Tool Bar

        openCfg = QPushButton(self)
        openCfg.setText("Open Config")
        openCfg.move(5, 30)
        openCfg.clicked.connect(self.openFractalConfig)

        saveCfg = QPushButton(self)
        saveCfg.setText("Save Config")
        saveCfg.move(openCfg.width() + 10, 30)
        saveCfg.clicked.connect(self.saveFractalConfig)

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

        fractalTypes = ["Mandelbrot", "Tricorn", "Burning Ship", "Perpendicular Mandelbrot", "Perpendicular Burning Ship", "Celtic Mandelbrot", "Celtic Mandelbar", "Perpendicular Celtic", "Buffalo", "Perpendicular Buffalo", "Mandelship", "Mandelbrot Heart", "Buffalo Heart", "Lamda Mandelbrot / Logistic Map", "Custom"]

        fractalType = QComboBox()
        fractalType.addItems(fractalTypes)

        fractalType.currentIndexChanged.connect(self.changeFractal)

        Power = QDoubleSpinBox()
        Power.setSingleStep(0.01)
        Power.setDecimals(10)
        Power.setValue(2)
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

        Coordinates = QHBoxLayout()
        CoordX = QLineEdit()
        CoordX.setText(str(offsetX128[0] + offsetX128[1]))
        CoordY = QLineEdit()
        CoordY.setText(str(offsetY128[0] + offsetY128[1]))
        Coordinates.addWidget(CoordX)
        Coordinates.addWidget(CoordY)

        StartRe.textChanged.connect(self.changeStartP)
        StartIm.textChanged.connect(self.changeStartP)

        iterations = QLineEdit()
        iterations.setValidator(QIntValidator())
        iterations.setText(str(itr))
        iterations.textChanged.connect(self.changeIter)

        Zoom = QLineEdit()
        Zoom.setValidator(QDoubleValidator())

        self.timer = QTimer()
        self.timer.setInterval(100)
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

        if curFractal != 14:
            formulaBtn.setEnabled(False)

        tab1.layout.addRow("Fractal", fractalType)
        tab1.layout.addRow("Power", Power)
        tab1.layout.addRow("Seed", StartingP)
        tab1.layout.addRow("Coordinates", Coordinates)
        tab1.layout.addRow("Iterations", iterations)
        tab1.layout.addRow("Zoom", Zoom)
        tab1.layout.addRow("Julia Set", juliaTrigger)
        tab1.layout.addRow(formulaBtn)
        # tab1.layout.addRow("Z Layer", zlay)

        tab1.setLayout(tab1.layout)

        # Tab 2 (Rendering)

        precision = QComboBox()
        precision.addItem("Double")
        precision.addItem("Quadruple (emulated)")

        precision.currentIndexChanged.connect(self.changePrec)

        colorMethod = QComboBox()
        colorMethod.addItem("Smooth iteration")
        colorMethod.addItem("Discrete iteration")

        colorMethod.currentIndexChanged.connect(self.coloringMethod)

        bailout = QLineEdit()
        bailout.setValidator(QDoubleValidator())
        bailout.setText(str(escRad))

        bailout.textChanged.connect(self.changeBail)

        useSlopes = QCheckBox()
        useSlopes.setChecked(False)
        useSlopes.clicked.connect(self.toggleSlopes)

        tab2.layout.addRow("Float Type", precision)
        tab2.layout.addRow("Iteration count", colorMethod)
        tab2.layout.addRow("Bailout", bailout)
        tab2.layout.addRow("Slopes", useSlopes)

        tab2.setLayout(tab2.layout)

        # Tab 3 (Gradient)

        col1 = QHBoxLayout()
        col2 = QHBoxLayout()
        col3 = QHBoxLayout()
        col4 = QHBoxLayout()
        col5 = QHBoxLayout()

        self.R1 = QSlider(Qt.Horizontal)
        self.R1.setMaximum(100)
        self.R1.setMinimum(0)

        self.R1.valueChanged.connect(self.editGradient)

        G1 = QSlider(Qt.Horizontal)
        G1.setMaximum(100)
        G1.setMinimum(0)

        G1.valueChanged.connect(self.editGradient)

        B1 = QSlider(Qt.Horizontal)
        B1.setMaximum(100)
        B1.setMinimum(0)

        B1.valueChanged.connect(self.editGradient)

        self.R1.setValue(col1R)
        G1.setValue(col1G)
        B1.setValue(col1B)

        col1.addWidget(self.R1)
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

        R5 = QSlider(Qt.Horizontal)
        R5.setMaximum(100)
        R5.setMinimum(0)

        R5.valueChanged.connect(self.editGradient)

        G5 = QSlider(Qt.Horizontal)
        G5.setMaximum(100)
        G5.setMinimum(0)

        G5.valueChanged.connect(self.editGradient)

        R5.setValue(col5R)
        G5.setValue(col5G)

        col5.addWidget(R5)
        col5.addWidget(G5)

        # Presets

        presCol = QComboBox()
        presCol.addItem("Default")
        presCol.addItem("Fire")
        presCol.addItem("Sunset")
        presCol.addItem("Pastel Rainbow")
        presCol.addItem("UniFract I")
        presCol.addItem("Binary")

        presCol.currentIndexChanged.connect(self.changePresets)

        # Random Colors

        randCol = QPushButton()
        randCol.setText("Random Colors")
        randCol.clicked.connect(self.randomColors)

        # Interior Color

        inColorBtn = QPushButton('Interior Color')
        inColorBtn.clicked.connect(self.inColor)

        tab3.layout.addRow("Presets", presCol)
        tab3.layout.addRow("Color 1", col1)
        tab3.layout.addRow("Color 2", col2)
        tab3.layout.addRow("Color 3", col3)
        tab3.layout.addRow("Color 4", col4)
        tab3.layout.addRow("Color 5", col5)
        tab3.layout.addRow(randCol)
        tab3.layout.addRow(inColorBtn)

        tab3.setLayout(tab3.layout)

        # Tab 4 (Other)

        perturb = QCheckBox()
        perturb.setChecked(False)
        perturb.clicked.connect(self.setPerturbation)

        perturb.setToolTip("Slower, but allows for deeper zooming.")

        tab4.layout.addRow("Use Perturbation", perturb)

        tab4.setLayout(tab4.layout)

    def inColor(self):
        global inColorR, inColorG, inColorB
        color = QColorDialog.getColor()

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        if color.isValid():
            inColorR = int(color.getRgb()[0])
            inColorG = int(color.getRgb()[1])
            inColorB = int(color.getRgb()[2])

        self.updateThread.start()

        gl.glUniform3f(gl.glGetUniformLocation(program, "inColor"), inColorR/255, inColorG/255, inColorB/255)

    def openWindow(self):
        global program
        self.fractalEditor.show()
        self.glWidget.update()

    def changeFractal(self):
        global fractalType
        global program
        global curFractal
        global formulaBtn

        #program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        curFractal = fractalType.currentIndex()

        thread("Update Thread", self.glWidget.update).start()

        if curFractal == 14:
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
        global StartRe
        global StartIm

        #program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        thread("Update Thread", self.glWidget.update).start()

        isJulia = juliaTrigger.isChecked()

        if juliaTrigger.isChecked():
            StartX = -1.0
            StartY = -0.5
        else:
            StartX, StartY = 0.0, 0.0

        gl.glUniform1f(gl.glGetUniformLocation(program, "juliaEnabled"), isJulia)

    def changePrec(self):
        global precision
        global useQuadPrec

        if precision.currentIndex() == 0:
            useQuadPrec = False
        elif precision.currentIndex() == 1:
            useQuadPrec = True

        self.glWidget.update()
        gl.glUniform1f(gl.glGetUniformLocation(program, "useQuadPrec"), useQuadPrec)

    def changePower(self):
        global power
        global Power
        global program

        #program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        thread("Update Thread", self.glWidget.update).start()

        power = float(Power.value())
        gl.glUniform1f(gl.glGetUniformLocation(program, "POWER"), power)

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

        #program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        thread("Update Thread", self.glWidget.update).start()

        if StartRe.text() != "" and StartIm.text() != "":
            StartX = float(StartRe.text())
            StartY = float(StartIm.text())

            gl.glUniform2d(gl.glGetUniformLocation(program, "Start"), StartX, StartY)

    def changeIter(self):
        global iterations
        global itr
        global program

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        if (iterations.text() != ""):
            itr = int(iterations.text())

        gl.glUniform1i(gl.glGetUniformLocation(program, "itr"), itr)

        thread("Update Thread", self.glWidget.update).start()

    def info(self):
        global Zoom
        global zoom
        global CoordX
        global CoordY

        Zoom.setText(str(zoom))
        CoordX.setText(str(offsetX128[0] + offsetX128[1]))
        CoordY.setText(str(offsetY128[0] + offsetY128[1]))

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
        global R5
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

        #program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        try:
            col1R = int(self.R1.value())
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
            col5R = int(R5.value())

            gl.glUniform3f(gl.glGetUniformLocation(program, "col1"), col1R / 100, col1G / 100, col1B / 100)
            gl.glUniform3f(gl.glGetUniformLocation(program, "col2"), col2R / 100, col2G / 100, col2B / 100)
            gl.glUniform3f(gl.glGetUniformLocation(program, "col3"), col3R / 100, col3G / 100, col3B / 100)
            gl.glUniform3f(gl.glGetUniformLocation(program, "col4"), col4R / 100, col4G / 100, col4B / 100)
            gl.glUniform3f(gl.glGetUniformLocation(program, "col5"), col5R / 100, col5G / 100, col5B / 100)
        except:
            e = sys.exc_info()
            self.error = errWindow(f"""
<P><font>
{str(e[0].__name__)}: {str(e[1])}
<br>
</br>
<br>
</br>
Traceback: {"None" if str(e[2]) == None else str(e[2])}
<br>
</br>
<br>
</br>
<u><font color="#ff0000">
UniFract II will try to continue, but there might be more problems in the future.
</font></u>
</font></P>""")

        thread("Update Thread", self.glWidget.update).start()

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
        global R5

        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        col1R = random.randint(0, 100)
        col1G = random.randint(0, 100)
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

        thread("Update Thread", self.glWidget.update).start()

    def changePresets(self):
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
        global presCol
        global program

        #program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        if presCol.currentText() == "Default":
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

        elif presCol.currentText() == "Fire":
            col1R = 0
            col1G = 0
            col1B = 0
            col2R = 100
            col2G = 50
            col2B = 0
            col3R = 100
            col3G = 75
            col3B = 0
            col4R = 100
            col4G = 100
            col4B = 100
            col5R = 100
            col5G = 80
            col5B = 0
            col6R = 100
            col6G = 50
            col6B = 0
            col7R = 100
            col7G = 10
            col7B = 10
            col8R = 0
            col8G = 0
            col8B = 0
            col9R = 30
            col9G = 20
            col9B = 10
            col10R = 100
            col10G = 97
            col10B = 0

        elif presCol.currentText() == "Sunset":
            col1R = 0
            col1G = 0
            col1B = 40
            col2R = 50
            col2G = 0
            col2B = 72
            col3R = 100
            col3G = 50
            col3B = 1
            col4R = 100
            col4G = 100
            col4B = 65
            col5R = 100
            col5G = 100
            col5B = 100
            col6R = 0
            col6G = 0
            col6B = 40
            col7R = 50
            col7G = 0
            col7B = 72
            col8R = 100
            col8G = 50
            col8B = 1
            col9R = 100
            col9G = 100
            col9B = 65
            col10R = 100
            col10G = 100
            col10B = 100
        
        elif presCol.currentText() == "Pastel Rainbow":
            col1R = 100
            col1G = 40
            col1B = 38
            col2R = 99
            col2G = 69
            col2B = 26
            col3R = 99
            col3G = 99
            col3B = 59
            col4R = 62
            col4G = 87
            col4B = 62
            col5R = 62
            col5G = 75
            col5B = 81
            col6R = 80
            col6G = 60
            col6B = 78
            col7R = 100
            col7G = 40
            col7B = 38
            col8R = 99
            col8G = 69
            col8B = 26
            col9R = 99
            col9G = 99
            col9B = 59
        
        elif presCol.currentText() == "UniFract I":
            col1R = 56
            col1G = 0
            col1B = 60
            col2R = 18
            col2G = 27
            col2B = 100
            col3R = 18
            col3G = 90
            col3B = 100
            col4R = 3
            col4G = 100
            col4B = 23
            col5R = 91
            col5G = 100
            col5B = 3
            col6R = 100
            col6G = 23
            col6B = 3
            col7R = 56
            col7G = 11
            col7B = 0
            col8R = 56
            col8G = 0
            col8B = 60
            col9R = 18
            col9G = 27
            col9B = 100

        gl.glUniform3f(gl.glGetUniformLocation(program, "col1"), col1R / 100, col1G / 100, col1B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col2"), col2R / 100, col2G / 100, col2B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col3"), col3R / 100, col3G / 100, col3B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col4"), col4R / 100, col4G / 100, col4B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col5"), col5R / 100, col5G / 100, col5B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col6"), col6R / 100, col6G / 100, col6B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col7"), col7R / 100, col7G / 100, col7B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col8"), col8R / 100, col8G / 100, col8B / 100)
        gl.glUniform3f(gl.glGetUniformLocation(program, "col9"), col9R / 100, col9G / 100, col9B / 100)

        thread("Update Thread", self.glWidget.update).start()

    def changeBail(self):
        global program
        global escRad
        global bailout

        escRad = float(bailout.text())
        self.glWidget.update()
        gl.glUniform1d(gl.glGetUniformLocation(program, "bailout"), escRad)

    def coloringMethod(self):
        global colorMethod
        global smoothColoring
        global program

        if colorMethod.currentIndex() == 0:
            smoothColoring = True
        elif colorMethod.currentIndex() == 1:
            smoothColoring = False

        self.glWidget.update()

        gl.glUniform1f(gl.glGetUniformLocation(program, "smoothColoring"), smoothColoring)

    def toggleSlopes(self):
        global useSlopes
        global program
        global slopes
        slopes = useSlopes.isChecked()
        self.glWidget.update()
        gl.glUniform1f(gl.glGetUniformLocation(program, "slopesEnabled"), slopes)

    def exportImgs(self):
        global zoom
        global program
        #program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))
        frames = 0
        while zoom >= 100.0:
            self.glWidget.paintGL()
            image = self.glWidget.grabFrameBuffer()
            
            image.save("fr_%04d.png" % frames)

            gl.glUniform1d(gl.glGetUniformLocation(program, "zoom"), zoom)
            
            zoom /= 1.02
            frames += 1
            self.glWidget.update()

    def openFractalConfig(self):
        global zoom
        global offsetX
        global offsetY
        global fractalType
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Config", "", "UniFract II Config (*.uf2)", options=options)
        if fileName:
            f = open(fileName, "r").readlines()
            fractalType.setCurrentIndex(int(f[1]))
            zoom = float(f[4])
            offsetX = float(f[7])
            offsetY = float(f[8])
            print("Config loaded!")
            self.glWidget.update()

    def saveFractalConfig(self):
        global zoom
        global offsetX
        global offsetY
        global fractalType
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "UniFract II Config (*.uf2)", options=options)
        if fileName:
            f = open(fileName,"w")
            f.write(f"""Fractal Type
{fractalType.currentIndex()}

Zoom
{zoom}

Coordinates
{offsetX}
{offsetY}""")
            print("File", fileName, "saved successfully!")
            f.close()

    def setPerturbation(self):
        global perturb
        global perturbationEnabled
        global fractalType
        global Power
        global juliaTrigger
        
        program = compileProgram(compileShader(vertexShader, gl.GL_VERTEX_SHADER), compileShader(fragmentShader, gl.GL_FRAGMENT_SHADER))

        thread("Update Thread", self.glWidget.update).start()

        perturbationEnabled = perturb.isChecked()

        if perturb.isChecked():
            #fractalType.setCurrentIndex(0)
            #fractalType.setEnabled(False)
            Power.setEnabled(False)
            juliaTrigger.setEnabled(False)
        else:
            #fractalType.setEnabled(True)
            Power.setEnabled(True)
            juliaTrigger.setEnabled(True)

        gl.glUniform1f(gl.glGetUniformLocation(program, "perturbationEnabled"), perturb.isChecked())

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

app = QApplication(sys.argv)

win = MainWindow()
win.show()

print(threading.active_count())

sys.exit(app.exec_())
