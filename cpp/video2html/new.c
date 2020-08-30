#include "head.h"

const char chapter[95][20] = {
"第零一集", "第零二集", "第零三集", "第零四集",
"第零五集", "第零六集", "第零七集", "第零八集",
"第零九集", "第一十集", "第一一集", "第一二集",
"第一三集", "第一四集", "第一五集", "第一六集",
"第一七集", "第一八集", "第一九集",
"第二十集", "第二一集", "第二二集", "第二三集",   "第二四集", "第二五集", "第二六集", "第二七集",  "第二八集", "第二九集",
"第三十集", "第三一集", "第三二集", "第三三集",   "第三四集", "第三五集", "第三六集", "第三七集",  "第三八集", "第三九集",
"第四十集", "第四一集", "第四二集", "第四三集",   "第四四集", "第四五集", "第四六集", "第四七集",  "第四八集", "第四九集",
"第五十集", "第五一集", "第五二集", "第五三集",   "第五四集", "第五五集", "第五六集", "第五七集",  "第五八集", "第五九集",
"第六十集", "第六一集", "第六二集", "第六三集",   "第六四集", "第六五集", "第六六集", "第六七集",  "第六八集", "第六九集",
"第七十集", "第七一集", "第七二集", "第七三集",   "第七四集", "第七五集", "第七六集", "第七七集",  "第七八集", "第七九集",
"第八十集", "第八一集", "第八二集", "第八三集",   "第八四集", "第八五集", "第八六集", "第八七集",  "第八八集", "第八九集",
"第九十集", "第九一集", "第九二集", "第九三集"};

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
    char dir[20]      = "blenderVideos";
    int len = 93;
    char title[60] = "blender";
    getIndexHtml(len,dir,title);
    char src[120]      = "  src=\"blender";

    char command[30];
    sprintf(command,"mkdir %s",dir);
    system(command);
    for(int j = 1;j <= len;j++)
    {
        writeToFileNew(j,"mp4",dir,src);
    }
}
/*}}}*/
