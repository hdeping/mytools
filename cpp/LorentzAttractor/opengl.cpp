/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-17 21:16:21
    @project      : draw the lorentz attractor
    @version      : 1.0
    @source file  : opengl.cpp

============================

*/


#include <opengl.h>

/*void graphic::init{{{*/
void graphic::init()
{
    glClearColor(0.3,0.9,0.5,1.0);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_LINE_SMOOTH_HINT,GL_DONT_CARE);
    glHint(GL_POINT_SMOOTH_HINT,GL_DONT_CARE);
    glLineWidth(2.0);
    glPointSize(2.0);
}
/*}}}*/
/*void graphic::display{{{*/
void graphic::display(void){
    // Called when opengl needs 
    // to update the display
    glClear( GL_COLOR_BUFFER_BIT);  
    glLoadIdentity();
    gluLookAt(0.0,0.0,0.5,0.0,0.0,0.0,0.0,1.0,0.0);
    glColor3f(0.0,0.0,1.0);
    // draw lines
    glBegin (GL_LINES);
    for(int i = 0;i < lineNum;i++)
    {
        glVertex3f(xcoor[i][0],xcoor[i][1],0.0);
        glVertex3f(xcoor[i+1][0],xcoor[i+1][1],0.0);
    }
    glEnd();
    
    glFlush();
    glutSwapBuffers();
}
/*}}}*/
/*void graphic::keyboard{{{*/
void graphic::keyboard(unsigned char key, int x,int y){
    if ( key == 27 || key == 'q'    ){
        exit(0); // 27 is the "Escape" key
    }
}
/*}}}*/
/*void graphic:: reshape{{{*/
void graphic:: reshape(int width ,int height){
    /*
     * called when the window is created, moved or resized
     * */
    glViewport(0,0,(GLsizei)width,(GLsizei)height);
    glMatrixMode(GL_PROJECTION);  // select projection matrix
    glOrtho(- 1.0,1.0,- 1.0,1.0, - 1.0,1.0);  // the unit cube
}
/*}}}*/
