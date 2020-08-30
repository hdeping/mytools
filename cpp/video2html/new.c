#include "head.h"


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
        sprintf(htmlName,"files/%s/%d.html",dir,i);
        sprintf(contentName,"Chapter %d",i);

        printf("%s\n",htmlName);
        printf("%s\n",contentName);
        fprintf(fp,"<a href=\"%s\">\n",htmlName);
        fprintf(fp,"    <div style=\"position:relative\">\n");
        fprintf(fp,"    <img src = \"buliangren.jpg\" width = \"200\" height = \"120\" />\n");
        fprintf(fp,"    <div style=\"position:absolute;z-index:8;left:100px;top:100px\">\n");
        fprintf(fp, "%s</div> </div>\n",contentName);
    }
    fprintf(fp,"  </body>\n");
    fprintf(fp,"</html>\n");
    fclose(fp);
}
/*}}}*/
/*int main{{{*/
int main( int argc,char *argv[]){
    char dir[20]      = "gallery";
    int len = 9;
    char title[60] = "复杂网络绘图";
    getIndexHtml(len,dir,title);
}
/*}}}*/
