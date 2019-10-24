#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <assert.h>
#include <GL/glut.h>

/*namespace graphic{{{*/
namespace graphic
{
    const int cy_times = (int)1E6;
    const int freq = (int)3E3;
    void init();
    void display(void);
    void keyboard(unsigned char key, int x,int y);
    void reshape(int width ,int height);
};
/*}}}*/

