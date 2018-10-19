/* 
FILTER.C 
An ANSI C implementation of MATLAB FILTER.M (built-in)
Written by Chen Yangquan <elecyq@nus.edu.sg> 1998-11-11
test main program and commented is writen by Group13 
- Danang University of Technology, VietNam
tngotran@gmail.com

Example of FILTER.C that translate from filter.m matlab
Y = filter(B,A,X) 
 a(1)*y(n) = b(1)*x(n) + b(2)*x(n-1) + ... + b(nb+1)*x(n-nb)
							- a(2)*y(n-1) - ... - a(na+1)*y(n-na)
							
							
//this program is correspoding to command "y = filter(b,1,x)" in matlab
#include<stdio.h>
#define ORDER 2  //ORDER is the number of element a[] or b[] minus 1(array a[] and b[] are coefficents of filter Dirac II)
#define NP 11 //NP is the number of output or input filter minus 1;
void filter(int , float *, float *, int , float *, float *);
void main(void){
	
	float a[3]={1,0,0};
	float b[3]={1,2,3};
  	float x[12]={1,2,3,4,8,7,6,5,1,4,2,3};
  	float y[12];
  	
	filter(ORDER,a,b,NP,x,y);
	
	return;
}
*/

void filter(int ord, float *a, float *b, int np, float *x, float *y,int st)
	{
    int i,j;
	y[st]=b[0]*x[st];
	for (i=1;i<ord+1;i++)
	{
        y[st+i]=0.0;
        for (j=0;j<i+1;j++)
        	y[st+i]=y[st+i]+b[j]*x[st+i-j];
        for (j=0;j<i;j++)
        	y[st+i]=y[st+i]-a[j+1]*y[st+i-j-1];
        
	}
	/* end of initial part */
	for (i=ord+1;i<np+1;i++)
	{
		y[st+i]=0.0;
			for (j=0;j<ord+1;j++)
				y[st+i]=y[st+i]+b[j]*x[st+i-j];
			for (j=0;j<ord;j++)
				y[st+i]=y[st+i]-a[j+1]*y[st+i-j-1];
	}
	return;
}
