/*

============================

    @author       : Deping Huang
    @mail address : xiaohengdao@gmail.com
    @date         : 2020-09-17 11:48:44
    @project      : my common cpp tools
    @version      : 1.0
    @source file  : MyCommon.c

============================

*/
#include "MyCommon.h"

int getSum(int *array,int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i)
    {
        sum += array[i];
    }
    return sum;
}
void printArray(int *array,int begin,int end) {
    for (int i = begin; i < end; ++i)
    {
        printf("i = %d,item = %d\n", i,array[i]);
    }
    return ;
}
void runOccupy() {
    for (int i = 0; i < N; ++i) {
       data[i] = 0;
    }

    srand(time(NULL));
    int freq = 1000;
    int pos;
    for (int i = 0; i < N; ++i) {
        pos = random()%N;
        if (data[pos] == 1) {
            continue;
        }
        else{
            switch (pos) {
                case 0:
                    if (data[1] == 0) {
                        data[pos] = 1;
                    }
                    break;
                case N-1:
                    if (data[N-2] == 0) {
                        data[pos] = 1;
                    }
                    break;
                default:
                    if (data[pos-1] == 0 && data[pos+1] == 0)
                    {
                        data[pos] = 1;
                    }
                    
            }
        }

    }
    printf("sum = %d\n", getSum(data,N));
    printArray(data,0,10);
    return ;
}
