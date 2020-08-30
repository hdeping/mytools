#include "head.h"


/*void writeToFile{{{*/
void writeToFile(int i,int j,char *filetype,char *dir,char *src)
{
    FILE *fp;
    char filename[20];
    int num;
    if ( i < 10 )
    {
        if ( j < 10 )
        {
            num = 0;
        }
        else
        {
            num = 1;
        }
        
    }
    else
    {
        if ( j < 10 )
        {
            num = 2;
        }
        else
        {
            num = 3;
        }
    }
    switch(num)
    {
        case 0:
            sprintf(filename,"%s/0%d0%d.html",dir,i,j);
            break;
        case 1:
            sprintf(filename,"%s/0%d%d.html",dir,i,j);
            break;
        case 2:
            sprintf(filename,"%s/%d0%d.html",dir,i,j);
            break;
        default:
            sprintf(filename,"%s/%d%d.html",dir,i,j);
            break;
    }
    fp= fopen(filename,"w");
    assert(fp != NULL);
    fprintf(fp,"<video             \n");
    switch(num)
    {
        case 0:
            fprintf(fp,"%s0%d/0%d.%s\"    \n",src,i,j,filetype);
            break;
        case 1:
            fprintf(fp,"%s0%d/%d.%s\"    \n",src,i,j,filetype);
            break;
        case 2:
            fprintf(fp,"%s%d/0%d.%s\"    \n",src,i,j,filetype);
            break;
        default:
            fprintf(fp,"%s%d/%d.%s\"    \n",src,i,j,filetype);
            break;
    }

    fprintf(fp,"  volume=\"20\"      \n");
    fprintf(fp,"  controls         \n");
    fprintf(fp,"  autoplay         \n");
    fprintf(fp,"  width=\"720\"      \n");
    fprintf(fp,"  height=\"480\"     \n");
    fprintf(fp,"  type=\"video/mp4\">\n");
    fprintf(fp,"</video>           \n");
    fclose(fp);
}
/*}}}*/
/*void writeToFileNew{{{*/
void writeToFileNew(int j,char *filetype,char *dir,char *src)
{
    FILE *fp;
    char filename[20];
    int num;
    if ( j < 10 )
    {
        num = 0;
    }
    else if(j < 100)
    {
        num = 1;
    }
    else
    {
        num = 2;
    }
    switch(num)
    {
        case 0:
            sprintf(filename,"%s/00%d.html",dir,j);
            break;
        case 1:
            sprintf(filename,"%s/0%d.html",dir,j);
            break;
        default:
            sprintf(filename,"%s/%d.html",dir,j);
            break;
    }
    fp= fopen(filename,"w");
    assert(fp != NULL);
    fprintf(fp,"<video             \n");
    switch(num)
    {
        case 0:
            fprintf(fp,"%s/0%d.%s\"    \n",src,j,filetype);
            break;
        default:
            fprintf(fp,"%s/%d.%s\"    \n",src,j,filetype);
            break;
    }

    fprintf(fp,"  volume=\"20\"      \n");
    fprintf(fp,"  controls         \n");
    fprintf(fp,"  autoplay         \n");
    fprintf(fp,"  width=\"720\"      \n");
    fprintf(fp,"  height=\"480\"     \n");
    fprintf(fp,"  type=\"video/mp4\">\n");
    fprintf(fp,"</video>           \n");
    fclose(fp);
}
/*}}}*/
/*void getIndexHtml{{{*/
void getIndexHtml(int line,char *dir,char *title)
{
    FILE *fp;
    char filename[20];
    sprintf(filename,"%s.html",dir);
    fp= fopen(filename,"w");
    assert(fp != NULL);

    fprintf(fp,"<meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"> \n");
    fprintf(fp,"  <h1 style=\"text-align:center\" >\n");
    fprintf(fp,"      <font color=\"blue\">\n");
    fprintf(fp,"%s\n",title);
    fprintf(fp,"  </h1>\n");
    fprintf(fp,"<body background=\"zhuxian.jpg\">\n");
    char tmp[40] = ".html\"><img src = \"cloud.png\"/>";
    for(int i = 1;i <= line;i++)
    {
        if ( i < 10 )
        {
            fprintf(fp,"    <a href=\"%s/00%d%s00%d</a> \n",
                    dir,i,tmp,i);
        }
        else if(i < 100)
        {
            fprintf(fp,"    <a href=\"%s/0%d%s0%d</a> \n",
                    dir,i,tmp,i);
        }
        else
        {
            fprintf(fp,"    <a href=\"%s/%d%s%d</a> \n",
                    dir,i,tmp,i);
        }
        if ( i % 5 == 0  )
        {
            fprintf(fp,"<br>\n");
        }
    }
    fprintf(fp,"  </body>\n");
    fprintf(fp,"</html>\n");
    fclose(fp);
}
/*}}}*/
/*int main{{{*/
int main( int argc,char *argv[]){
    // int num[14] = {36,25,17,4,36,29,16,47,33,9,21,15,40,55};
    char src[40]      = "  src=\"files/c_XiaoJiaYu";
    char filetype[10] = "mp4";
    char dir[20]      = "xiaojiayu";
    int len = 102;
    /*
     * 
    for(int j = 1;j <= len;j++)
    {
        writeToFileNew(j,filetype,dir,src);
    }
     * */
    char title[40] = "小甲鱼  算法与数据结构";
    getIndexHtml(len,dir,title);
    /**
     * 
    char command[40];
    sprintf(command,"mkdir %s",dir);
    system(command);
    for(int j = 1;j <= len;j++)
    {
        writeToFileNew(j,filetype,dir,src);
    }
     * */
}
/*}}}*/
