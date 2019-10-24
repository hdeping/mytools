/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-17 21:18:13
    @project      : draw the lorentz attractor
    @version      : 1.0
    @source file  : Lorentz.h

============================

*/


#include "head.h"
#define N 10000
#define freq 1
using namespace graphic;
double sigma = 0.5;
extern double xcoor[N][3];
extern double arrDraw[10][2];
extern int lineNum;
int num = 0;

typedef struct tagBITMAPFILEHEADER {  
    unsigned char bfType[2];  
    unsigned char bfSize[4];  
    unsigned char bfReserved1[2];  
    unsigned char bfReserved2[2];  
    unsigned char bfOffBits[4];  
} BITMAPFILEHEADER;  
typedef struct tagBITMAPINFOHEADER {  
    unsigned char biSize[4];  
    unsigned char biWidth[4];  
    unsigned char biHeight[4];  
    unsigned char biPlanes[2];  
    unsigned char biBitCount[2];  
    unsigned char biCompression[4];  
    unsigned char biSizeImage[4];  
    unsigned char biXPelsPerMeter[4];  
    unsigned char biYPelsPerMeter[4];  
    unsigned char biClrUsed[4];  
    unsigned char biClrImportant[4];  
} BITMAPINFOHEADER;  
BITMAPFILEHEADER fileHeader;  
BITMAPINFOHEADER infoHeader;  
GLfloat ctlpoints[4][4][3];  
GLUnurbsObj *theNurb; 