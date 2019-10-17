/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2019-10-17 21:14:44 (modified)
    @project      : draw the lorentz attractor
    @version      : 1.0
    @source file  : Lorentz.cpp

============================

*/

#include <Lorentz.h> 


/*void LongToByte{{{*/
void LongToByte(unsigned long value,unsigned char * outResult){  
    int i=0;  
    for(i=0;i<4;i++){  
        outResult[i] = (unsigned char)(value%16);  
        value = value/16;  
        outResult[i] += (unsigned char)(value%16)*16;  
        value = value/16;  
    }  
    outResult[4] = '\0';  
}  
/*}}}*/
/*void bmpWriter{{{*/
void bmpWriter()
{
    num++;
    char filename[20];
    char pngname[20];
    char bmpname[20];
    if ( num < 10 )
    {
        sprintf(filename,"000%d.",num);
    }
    else if ( num < 100 )
    {
        sprintf(filename,"00%d.",num);
    }
    else if ( num < 1000)
    {
        sprintf(filename,"0%d.",num);
    }
    else
    {
        sprintf(filename,"%d.",num);
    }
    sprintf(bmpname,"%sbmp",filename);
    sprintf(pngname,"%spng",filename);
    
    
    printf("Write to file: %s\n",pngname);
    GLint viewPort[4] = { 0 };  
    glGetIntegerv(GL_VIEWPORT, viewPort);  
    GLbyte * buffer =  
            (GLbyte *)malloc(viewPort[2] * viewPort[3] * sizeof(GLbyte) * 3);  
    glReadPixels(viewPort[0], viewPort[1], viewPort[2], viewPort[3],  
            GL_BGR, GL_UNSIGNED_BYTE, buffer);  
    long fileSize = viewPort[2] * viewPort[3] * 3 + 54;  
    //int i=0;  
    fileHeader.bfType[0] = 0x42;  
    fileHeader.bfType[1] = 0x4d;  
    LongToByte(fileSize, fileHeader.bfSize);  
    LongToByte(54, fileHeader.bfOffBits);  
    LongToByte(sizeof(infoHeader), infoHeader.biSize);  
    LongToByte(viewPort[2], infoHeader.biWidth);  
    LongToByte(viewPort[3], infoHeader.biHeight);  
    infoHeader.biPlanes[0] = 0x01;  
    infoHeader.biPlanes[1] = 0x00;  
    infoHeader.biBitCount[0] = 0x18;  
    infoHeader.biBitCount[1] = 0x00;  
    LongToByte(0, infoHeader.biCompression);  
    LongToByte((viewPort[2] * viewPort[3]), infoHeader.biSizeImage);  

    FILE * fp = fopen(bmpname, "w+");  
    fwrite(&fileHeader, sizeof(fileHeader), 1, fp);  
    fwrite(&infoHeader, sizeof(infoHeader), 1, fp);  
    fwrite(buffer, 1, (viewPort[2] * viewPort[3] * 3), fp);  
    fclose(fp);  
    free(buffer);  
    char command[80];
    sprintf(command,"convert %s %s",bmpname,pngname);
    system(command);
    sprintf(command,"rm %s",bmpname);
    system(command);
}
/*}}}*/
/*void evolution{{{*/
void evolution()
{
    /**
     * 
    if ( argc == 1  )
    {
        printf("Please input a file\n");
        printf("For Example: '<command> <data.file>'\n");
        return 0;
    }
    double c = atof(argv[1])/10.0;
     * */
    
    int i=0;
    double x0,y0,z0,x1,y1,z1;
    double h = 0.01;
    double a = 10.0;
    double b = 28.0;
    // double c = 8.0 / 3.0;
    double c = 2.8;

    x0 = 0.1;
    y0 = 0;
    z0 = 0;

    double max[3] = {0.0};
    double coor[3];
    for (i=0;i<N;i++) {
       coor[0] = x0;
       coor[1] = y0;
       coor[2] = z0;
       x1 = x0 + h * a * (y0 - x0);
       y1 = y0 + h * (x0 * (b - z0) - y0);
       z1 = z0 + h * (x0 * y0 - c * z0);
       x0 = x1;
       y0 = y1;
       z0 = z1;
       for(int i = 0;i < 3;i++)
       {
           if ( abs(coor[i]) > max[i] )
           {
               max[i] = abs(coor[i]);
           }
       }
       xcoor[i][0] = x0;
       xcoor[i][1] = z0;
       
    }
    
    // normalization
    for(int i = 0;i < N;i++)
    {
       xcoor[i][0] /= max[0]; 
       xcoor[i][1] /= max[2]; 
       xcoor[i][1] -= 0.5; 
    }
    // display

    for(int i = 0;i < N - 10;i++)
    {
       for(int j = 0;j < 10;j++)
       {
           for(int k = 0;k < 2;k++)
           {
               arrDraw[j][k] = xcoor[i+j][k];
           }
       }
       if ( i%2 == 0  )
       {
            lineNum = i;
            // printf("i = %d\n",i);
            display();
            bmpWriter();
            if ( num == 4900)
            {
                exit(0);
            }
            
       }
    }

    return ;
}
/*}}}*/
void idle(void)
{
    evolution();
}
void draw(int argc,char *argv[])
{
    glutInit(&argc,argv);     // Initialise the opengl
    glutInitDisplayMode(GLUT_RGB|GLUT_DOUBLE|GLUT_MULTISAMPLE);
    glutInitWindowPosition(300,0);
    glutInitWindowSize(800,800);
    glutCreateWindow("Xiaohengdao");  // create the window

    init();

    glutDisplayFunc(display); // register the "display" funcion
    glutReshapeFunc(reshape);
    glutKeyboardUpFunc(keyboard);  // register the "keyboard" funcion
    glutIdleFunc(idle);
    glutMainLoop();           // enter the opengl main loop
}

int main( int argc,char *argv[]){
   int i,j,k;
   draw(argc,argv);
   // evolution();
   return 0;
}

