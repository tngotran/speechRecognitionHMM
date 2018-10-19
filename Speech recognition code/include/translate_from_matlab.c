/* rdct(matrix)
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define M_PI 3.141592653589793238462643383279502884 
#define N  8
void main(){

    int i, j,k, R,C;
    float temp;
    float DCTMatrix[1][N];
    
    float Matrix[1][N]={ 
      {1,4,2,5,7,4,8,9}
     
     
     };
     R = 1;
     C = N;
    for (i = 0; i < R; i++) {
        for (j = 0; j < C; j++) {
        DCTMatrix[i][j] = 0;
            for ( k=0; k<C; k++) {
            		temp  = 10%2;
  					DCTMatrix[i][j] += Matrix[i][k]*cos((M_PI*(k+1./2.)*j)/C);
 			}
 			DCTMatrix[i][j] *= powf(2,0.5*(j+2-j%2)-j/2); 	
        }
    }  
    DCTMatrix[0][0] /= sqrt(2);
    return;
 }
 */
 
 /*
 */
 