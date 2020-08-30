#include "head.h"

const char chapter[20][20] = {
"第零一集", "第零二集", "第零三集", "第零四集",
"第零五集", "第零六集", "第零七集", "第零八集",
"第零九集", "第一十集", "第一一集", "第一二集",
"第一三集", "第一四集", "第一五集", "第一六集",
"第一七集", "第一八集", "第一九集" };

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
    else
    {
        num = 1;
    }
    switch(num)
    {
        case 0:
            sprintf(filename,"%s/0%d.html",dir,j);
            break;
        case 1:
            sprintf(filename,"%s/%d.html",dir,j);
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
        case 1:
            fprintf(fp,"%s/%d.%s\"    \n",src,j,filetype);
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

    fprintf(fp,"<style type=\"text/css\">\n");
    fprintf(fp,"div{word-wrap:break-word;word-break:normal;}\n");
    fprintf(fp,"div{float:left}\n");
    fprintf(fp,"</style>\n");
    fprintf(fp,"<body background=\"zhuxian.jpg\">\n");

    char tmp[40] = ".html\"><img src = \"cloud.png\"/>";
    for(int i = 1;i <= line;i++)
    {
        char htmlName[30];
        char contentName[30];
        if ( i < 10 )
        {
            sprintf(htmlName,"%s/0%d.html",dir,i);
        }
        else
        {
            sprintf(htmlName,"%s/%d.html",dir,i);
        }
        printf("%s\n",htmlName);
        printf("%s\n",chapter[i - 1]);
        fprintf(fp,"<a href=\"%s\">\n",htmlName);
        fprintf(fp,"    <div style=\"position:relative\">\n");
        fprintf(fp,"    <img src = \"buliangren.jpg\" width = \"200\" height = \"120\" />\n");
        fprintf(fp,"    <div style=\"position:absolute;z-index:8;left:100px;top:100px\">\n");
        fprintf(fp, "%s</div> </div>\n",chapter[i - 1]);
    }
    fprintf(fp,"  </body>\n");
    fprintf(fp,"</html>\n");
    fclose(fp);
}
/*}}}*/
/*int main{{{*/
int main( int argc,char *argv[]){
    char dir[20]      = "beauty3";
    int len = 12;
    char title[60] = "校花的贴身高手 第三季";
    getIndexHtml(len,dir,title);
    char src[120]      = "  src=\"files/beauty3";

    for(int j = 1;j <= len;j++)
    {
        writeToFileNew(j,"mp4",dir,src);
    }
}
/*}}}*/
