/***************************************************************************** 
 * Have a nice day :)
 *
 * speaker_recognition_train.c
 *
 * Main Program to Identify a Speaker.
 *
 * The aim of this project is to determine the identity of the speaker
 * from the speech sample of the speaker and the trained vectors.
 *
 * Trained vectors are derived from the speech sample of the speaker at
 * a different time.
 * 
 * First the input analog speech signal is digitized at 8KhZ Sampling
 * Frequency using the on board ADC (Analog to Digital Converter)
 * The Speech sample is stored in an one-dimensional array.
 * Speech signal's are quasi-stationary. It means that the 
 * speech signal over a very short time frame can be considered to be a
 * stationary. The speech signal is split into frames. Each frame consists
 * of 256 Samples of Speech signal and the subsequent frame will start from
 * the 100th sample of the previous frame. Thus each frame will overlap
 * with two other subsequent other frames. This technique is called
 * Framing. Speech sample in one frame is considered to be stationary.
 *
 * After Framing, to prevent the spectral lekage we apply windowing. 
 * Here  Hamming window with 256 co-efficients is used.
 *
 * Third step is to convert the Time domain speech Signal into Frequency
 * Domain using Discrete Fourier Transform. Here Fast Fourier Transform
 * is used.
 *
 * The resultant transformation will result in a signal beeing complex
 * in nature. Speech is a real signal but its Fourier Transform will be 
 * a complex one (Signal having both real and imaginary). 
 *
 * The power of the signal in Frequency domain is calculated by summing
 * the square of Real and Imaginary part of the signal in Frequency Domain.
 * The power signal will be a real one. Since second half of the samples
 * in the frame will be symmetric to the first half (because the speech signal
 * is a real one) we ignore the second half (second 128 samples in each frame)
 *
 * Triangular filters are designed using Mel Frequency Scale. These bank of 
 * filters will approximate our ears. The power signal is then applied to 
 * these bank of filters to determine the frequency content across each filter.
 * In our implementation we choose total number of filters to be 20.
 * These 20 filters are uniformly spaced in Mel Frequency scale between 
 * 0-4KhZ.
 *
 * After computing the Mel-Frequency Spectrum, log of Mel-Frequency Spectrum
 * is computed.
 *
 * Discrete Cosine Tranform of the resulting signal will result in the 
 * computation of the Mel-Frequency Cepstral Co-efficient.
 *
 * Euclidean distance between the trained vectors and the Mel-Frequency
 * Cepstral Co-efficients are computed for each trained vectors. The
 * trained vector that produces the smallest Euclidean distance will  
 * be identified as the speaker.
 *
 *
 * Written by Group13.M.A.D 09ECE DUT VietNam
 * 
 * 
 *
 ******************************************************************************/

/*****************************************************************************
 * Include Header Files
 ******************************************************************************/
#include <stdlib.h>

#include "dsk6713_aic23.h"
#include "dsk6713_dip.h"
#include "dsk6713_led.h"
Uint32 fs=DSK6713_AIC23_FREQ_16KHZ;
#define DSK6713_AIC23_INPUT_MIC 0x0015
Uint16 inputsource= DSK6713_AIC23_INPUT_MIC; // select input
#include <stdio.h>
#include <math.h>
#include <csl_gpio.h>
#include <csl_gpiohal.h>
#include <csl_irq.h>
//#include "block_dc.h" // Header file for identifying the start of speech signal
//#include "detect_envelope.h" // Header file for identfying the start of speech signal
#include "filter.h"
#include "code.h"
#include "word0.h"
#include "word1.h"
#include "word2.h"
#include "word3.h"
#include "word4.h"
#include "word5.h"
#include "word6.h"
#include "word7.h"
/*****************************************************************************
 * Definition of Variables
 *****************************************************************************/
#define Us_fs 16000;
#define PI 3.141592654
#define TWOPI	(2.0*PI)
#define Number_Of_Filters 20 // Number of Mel-Frequency Filters
#define column_length 256 // Frame Length of the one speech signal
#define row_length 140 // Total number of Frames in the given speech signal
#define half_of_column 129 // Total number of Frames in the given speech signal
#define real_mel_cof 13 // real number of mel coefficents
#define num_words 8 // number of words
#define leng_real_store real_mel_cof*3 // the real number of coefficent store to calculate
#define nuCalNoi 10
//define for hmm
#define nStates 6
#define nSymbols 128
#define N	2400000

/*****************************************************************************
 * Custom Structure Definition
 *****************************************************************************/

struct Hmm{
    int nstates;            /**< number of states in the HMM */
    int nsymbols;           /**< number of possible symbols */
    float *a;               /**< A matrix - state transition probabilities */
    float *b;               /**< B matrix - symbol output probabilities */
    float *pi;              /**< Pi matrix - initial state probabilities */
};
struct Obs{
    int length;            /**< number of states in the HMM */
    int *data;              /**< Pi matrix - initial state probabilities */
};

struct complex { 
	float real;
	float imag;
}; // Generic Structure to represent real and imaginary part of a signal

struct buffer {
	struct complex data[row_length][column_length];
}; // Structure to store the input speech sample

struct mfcc {
	float data[row_length][real_mel_cof*3];
}; // Structure to store the Mel-Frequency Co-efficients

/*****************************************************************************
 * Assigning the data structures to external memory
 *****************************************************************************/

#pragma DATA_SECTION(real_buffer,".EXTRAM")
struct buffer real_buffer; //real_buffer is used to store the input speech.

#pragma DATA_SECTION(coeff,".EXTRAM")
struct mfcc coeff; //coeff is used to store the Mel-Frequency Spectrum.


#pragma DATA_SECTION(mfcc_ct,".EXTRAM")
struct mfcc mfcc_ct; //mfcc_ct is used to store the Mel-Frequency Cepstral Co-efficients.

#pragma DATA_SECTION(in_hmm,".EXTRAM")
struct Hmm in_hmm;
#pragma DATA_SECTION(in_obs,".EXTRAM")
struct Obs in_obs;

                    //large buffer size 300 secs


#pragma DATA_SECTION(record,".EXTRAM")

GPIO_Handle gpio_handle;  /* handle para el GPIO */
/*****************************************************************************
 * Variable Declaration
 *****************************************************************************/

int gain;           /* output gain (Used during Play-Back */
int signal_status; /* Variable to detect speech signal */
int count; /* Variable to count */
int column; /* Variable used for incrementing column (Samples inside Frame)*/
int row; /* Variable used for incrementing row(Number of Frames)*/
int program_control; /* Variable to identify where the program is
							Example: program_control=0 means program is 
							capturing input speech signal
							program_control=1 means that program has finished
							capturing input and ready for processing. At this
							time the input speech signal is replayed back
							program_control=2 means program is ready for 
							idenitification. */
float mfcc_vector[Number_Of_Filters]; /* Variable to store the vector of the speech signal */
float abso[row_length];
int endd;
int speech_index[14];
float Ya[row_length];
int start,cont_chek,upda_noi;
int signal_status = 0;
float noise_check,tnoise;
long int reidx;
short record[N];
/*****************************************************************************
 * Function Declaration
 *****************************************************************************/


int min(int ,int );
float max(float ,float );
void fft (struct buffer *,int, int, float *); /* Function to compute Fast Fouruer Transform */
int speechDetect(int *,int , int , float *);
void four1(float *, int , int );
short playback(); /* Function for play back */
//void log_energy(struct mfcc *); /* Function to compute Log of Power Signal */
void mfcc_coeff(struct mfcc * , struct mfcc *,float *,int ,int); /* Function to compute MFCC */
//void mfcc_vect(struct mfcc * , float *); /* Funciton to compute MFCC Vector */
void reshape(float *, float (*out)[real_mel_cof], int , int ,int);/*Function to strech a rectangular matrix to a straight matrix*/
void stretch(float (*in)[real_mel_cof], float *, int ,int , int ,int,int );
void mel_freq_spectrum(struct buffer *, struct mfcc *,float *,int ,int );
void disteu(struct mfcc *,float (*y)[leng_real_store],int *,int,int);
double run_hmm_fo(struct Hmm *,struct Obs *);
void delay(double );
void record_sound(void);
//////////////////////////////////////////////
//float run_hmm_bwa( struct Hmm *, struct Obs *, int , float );
//float calc_alpha(struct Hmm *,struct Obs *, float *, float *);
//void calc_beta(struct Hmm *, struct Obs *, float *, float *);
//void calc_gamma_sum(    float *, float *, int , int , float *);
//void calc_xi_sum(   struct Hmm *, struct Obs *, float *, float *, float *);
//void estimate_a(float *, float *, int ,int ,  float *, float *, float *);
//void estimate_b(    struct Hmm *, struct Obs *, float *, float *, float *);
//void estimate_pi(float *, float *, int , float *);


/*upda_noi used to stop update "tnoise" when program write sample of frame 10
 * tnoise - noise of new record - updated at begining of new recording
 * Ya[] - array contain abs of input signal-generated from fft funtion
 * signal_status - return the current row_index from "framing_window" function
 * cont_chek - 1 mean, flush and continue cheking voice-starting
 * 			- 2 means, voice is stops, stop recording and begin recognization
 */
GPIO_Config gpio_config = {          
    0x00000000, // gpgc = Modo Passthrough de Interrupciones y control directo sobre GP0
    0x0000FFFF, // gpen = Todos los pines de GPIO de 0 a 15 habilitados
    0x00000000, // gdir = Todos los pines de GPIO como entradas 
    0x00000000, // gpval = Guarda el nivel lógico de los pines
    0x00000000, // gphm all interrupts disabled for io pins 
    0x00000000, // gplm all interrupts to cpu or edma disabled 
    0x00000000  // gppol -- default state */
}; 
interrupt void c_int11()  {           /* interrupt service routine */
	
	//short sample_data;
	short out_sample;
	int i,range;
	if ( program_control == 0 ) { /* Beginning of Capturing input speech */
		//if(DSK6713_DIP_get(0)==0 || DSK6713_DIP_get(3)==0){
			
			
			signal_status = framing_windowing(input_sample(), &real_buffer,start);
			
			if(signal_status == 10 && upda_noi==1){//this If updates noise(frames 0->9)
				tnoise =0;
				fft(&real_buffer,0,10,Ya);
				for(i = 0;i<10;i++)
					tnoise += Ya[i];
				upda_noi = 0;									
			}
			else if(signal_status %20==0 && signal_status>=20&& cont_chek==1){//This IF check voice begining, signal_status = 20 means we fft(frames 10->19)
				noise_check = 0;
				if(signal_status==20) range = 10;
				else range = 20;
				fft(&real_buffer,signal_status-range,signal_status,Ya);
				
				for(i = signal_status-range; i < signal_status;i++)
					noise_check += Ya[i];
				if(noise_check < 5*tnoise){ 
					if(signal_status==20) {start = 2;cont_chek = 1;}
					else cont_chek = 2;//cont_chek == 1 for case flush all data of frame 10 ->20 and program is again write signal input to real_buffer and check whether the voice exist
				}else{
					DSK6713_LED_off(1);DSK6713_LED_off(2);DSK6713_LED_off(3);
					DSK6713_LED_on(0); 
					cont_chek = 0;//cont_chek =0 for case the voice exist and donot check anymore
				}
			}else	start = 0;
			
			
		//}
		out_sample = 0;							/* Output Data */
		if (signal_status >= row_length || cont_chek == 2){//|| (DSK6713_DIP_get(0)&&signal_status>10)) {
			DSK6713_LED_off(0);
			program_control = 2;
			endd = signal_status;
			start = 1;		       /* Capturing input signal is done */
			upda_noi = 1;
			cont_chek=1;
		}
		if(signal_status%20!=0) cont_chek = 1;//turn "cont_chek" on for checking voice stop next 20 frame
		output_sample(out_sample);		/* play nothing */
	}
//	else if(program_control == 3){
//		//        if(reidx%500==0)
//        DSK6713_LED_on(3);           		//turn on LED#3
////        else DSK6713_LED_off(3);
//        record[reidx++] = input_left_sample(); 		//input data
//  	    if (reidx>5000000){
//  	        program_control = 4;
//  	        DSK6713_LED_off(3);
//  	        reidx = 0;
//  	    }
//	}else if(program_control==4){
//  	 	  output_left_sample(record[reidx++]);
//  	 	   if (reidx>5000000){
//  	        program_control = 2;
//  	        DSK6713_LED_off(3);
//  	        reidx = 0;
//  	    	}
//  	 }
	
	return;
}

void main()  {	/* Main Function of the program */
	
	float (*code_pt)[leng_real_store];
	
	double loglik[num_words];
	double max_dis;
	int idx;
	
/****************************************************************************
 * Declaring Local Variables
 *****************************************************************************/
	int i,j,k; /* Variable used for counters */
  	int st,ed;/* Variable used for Counters */
  	
  	int out[row_length];
	
/*****************************************************************************
 * Execution of functions start
 ******************************************************************************/
 	DSK6713_LED_init();
	DSK6713_DIP_init();
	DSK6713_init();    

    gpio_handle = GPIO_open( GPIO_DEV0, GPIO_OPEN_RESET );

    GPIO_config(gpio_handle,&gpio_config);
    GPIO_pinDirection(gpio_handle , GPIO_PIN0, GPIO_OUTPUT);
    GPIO_pinDirection(gpio_handle , GPIO_PIN1, GPIO_OUTPUT);
    GPIO_pinDirection(gpio_handle , GPIO_PIN2, GPIO_OUTPUT);
    GPIO_pinWrite(gpio_handle,GPIO_PIN0,0);
    GPIO_pinWrite(gpio_handle,GPIO_PIN1,0);
    GPIO_pinWrite(gpio_handle,GPIO_PIN2,0);
    
    
    
	comm_intr();   /* init DSK, codec, McBSP */
	start = 1;upda_noi = 1;
/******************************************************************************
 * Initializing Variables
 *****************************************************************************/
 while(1){
	
// 	start =2 ;
 	gain = 1;
	column = 0;
	row = 0;
	program_control = 0;
//	signal_status = 0;
	count = 0;				  
	//stages=8;	/* Total Number of stages in FFT = 8 */
//	for ( i=0; i < row_length ; i++ ) { /* Total Number of Frames */
//			//Ya[i]=0.0;
//  		for ( j = 0; j < column_length ; j++) { /* Total Number of Samples in a Frame */
//	  		real_buffer.data[i][j].real = 0.0; /* Initializing real part to be zero */
//	  		real_buffer.data[i][j].imag = 0.0; /* Initializing imaginary part to be zero*/
//		}
// 	}
//  	for ( i=0; i<row_length; i++) { /* Total Number of Frames */
//  		for ( j=0; j<Number_Of_Filters; j++) { /* Total Number of Filters */
//			coeff.data[i][j] = 0.0; /* Initializing the co-effecient array */
//			mfcc_ct.data[i][j] = 0.0; /* Initializing the array for storing MFCC */
//		}
//	} /* End of Initializing the variables to zero */
	code_pt = code;
/*****************************************************************************
* Begining of the execution of the functions.
*****************************************************************************/
	IRQ_nmiEnable();          			//enable NMI interrupt
	output_sample(0); 
	while(program_control == 0);      /* infinite loop For Receiving/capturing alone*/	
	
/* Compute FFT of the input speech signal after Framing and Windowing */
	//k = fft(&real_buffer,speech_index,20,endd,Ya);
	k = speechDetect(speech_index,0,endd,Ya);
	
	if(k>4) k = 4;
	j=0;
	while(j<1){
  	st=speech_index[j++];
	ed=speech_index[j++]; 

/* Compute Mel-Frequency Spectrum of the speech signal in Power Spectrum Form */
	mel_freq_spectrum(&real_buffer,&coeff,abso,st,ed);
/* Computation of Discrete Cosine Transform */
	mfcc_coeff(&mfcc_ct,&coeff,abso,st,ed);
	////////////////////////////////DETECT WORD/////////////////////////////////
	disteu(&mfcc_ct, code_pt, out,st,ed);
	
	in_obs.length = ed-st+1;
	in_obs.data = out;
	
	in_hmm.pi = pi;
	
		i=0;idx=0;
		max_dis = -9999999.9999;
		while(i<num_words){
			if(i==0){in_hmm.a = a0;				
					 in_hmm.b = b0;}
			else if(i==1){in_hmm.a = a1;
						  in_hmm.b = b1;}
			else if(i==2){in_hmm.a = a2;
						  in_hmm.b = b2;}
			else if(i==3){in_hmm.a = a3;
					 	  in_hmm.b = b3;}
			else if(i==4){in_hmm.a = a4;
						  in_hmm.b = b4;}
			else if(i==5){in_hmm.a = a5;
						  in_hmm.b = b5;}
			else if(i==6){in_hmm.a = a6;
						  in_hmm.b = b6;}
			else if(i==7){in_hmm.a = a7;
						  in_hmm.b = b7;}
			
			loglik[i] = run_hmm_fo(&in_hmm, &in_obs);		
			if(max_dis < loglik[i]) {max_dis = loglik[i]; idx = i;}
			
			i++;
		}
		
	
		if(idx == 0){
			DSK6713_LED_on(3);
			//printf(" music");//togle
			GPIO_pinWrite(gpio_handle,GPIO_PIN0,1);
			delay(500000);
			GPIO_pinWrite(gpio_handle,GPIO_PIN0,0);		
		}else if(idx == 1){
			DSK6713_LED_on(2);
			GPIO_pinWrite(gpio_handle,GPIO_PIN0,1);
			delay(500000);
			GPIO_pinWrite(gpio_handle,GPIO_PIN0,0);
		//	printf(" off");			
			
		}else if(idx == 2){
			DSK6713_LED_on(2);
			DSK6713_LED_on(3);
			
//			GPIO_pinWrite(gpio_handle,GPIO_PIN0,1);
//			delay(500000);
//			GPIO_pinWrite(gpio_handle,GPIO_PIN0,0);
//			IRQ_nmiEnable();          			//enable NMI interrupt
//			output_sample(0); 
//			program_control = 3;
//			while(program_control == 3);
//			IRQ_nmiEnable();          			//enable NMI interrupt
//			output_sample(0); 
//			while(program_control == 4);
		//	printf(" record");
		}else if(idx == 3){
			DSK6713_LED_on(1);
			//printf(" video");
		}else if(idx == 4){
			DSK6713_LED_on(1);
			DSK6713_LED_on(3);
		//	printf(" take");
		}else if(idx == 5){
			DSK6713_LED_on(1);
			DSK6713_LED_on(2);
//			printf(" photo");
		}else if(idx == 6){			
			DSK6713_LED_on(1);
			DSK6713_LED_on(2);
			DSK6713_LED_on(3);
			GPIO_pinWrite(gpio_handle,GPIO_PIN1,1);
			delay(500000);
			GPIO_pinWrite(gpio_handle,GPIO_PIN1,0);			
//			printf(" next");
		}else if(idx == 7){
			DSK6713_LED_on(0);
			GPIO_pinWrite(gpio_handle,GPIO_PIN2,1);
			delay(500000);
			GPIO_pinWrite(gpio_handle,GPIO_PIN2,0);		
//			printf(" back");
		}else{
			DSK6713_LED_on(0);
			DSK6713_LED_on(1);
			DSK6713_LED_on(2);
			DSK6713_LED_on(3);
		}
	
 	}
	printf("\n");
	program_control = 0;
	signal_status = 0;
 }
}
/* Function to Compute Fast Fourier Transform */
//ed in fft is "<" not "<=" like Mel or Cof methods
void delay(double nu){
	while(nu>0) nu--;		
}
void fft (struct buffer *input_data,int st,int ed, float *Ya) {/* Input speech Data, n = 2^m, m = total number of stages */
	
	
	int c,r,z;
	
	float temp[column_length*2+1];
	for ( r = st; r < ed; r++) { /* For every frame */		
		z=1;
		for(c=0;c<column_length;c++){
			temp[z++] = input_data->data[r][c].real;
			temp[z++]= 0.0;
		}
		four1(temp, column_length,1);
		Ya[r] = 0; z=1;
		for(c=0;c<column_length;c++){
			input_data->data[r][c].real = temp[z++];			
			input_data->data[r][c].imag = temp[z++];
			Ya[r] += sqrt(input_data->data[r][c].real*input_data->data[r][c].real+input_data->data[r][c].imag*input_data->data[r][c].imag);
		}
		Ya[r] /= column_length;		
	}
	return;
}					
//ed in speechDetect is "<" not "<=" like Mel or Cof methods
int speechDetect(int *speech_index,int st, int ed, float *Ya){
	int z,i,j,k;
	
	int speech_start,speech_end;
	float IMX = -999999999;
	float IMN = 999999999;
	
	for(z=st;z<ed;z++){
		if(Ya[z]>IMX) IMX=Ya[z];
		else if(Ya[z]<IMN) IMN = Ya[z];
	}
	z = 0;	speech_start=0;speech_end=0;i=st;j=0;
		if((0.03*(IMX - IMN) + IMN) < (4*IMX))
			IMX = (0.03*(IMX - IMN) + IMN);//ITL
		else
			IMX = (4*IMX);//ITL
				
		IMN = 5*IMX;//ITU		
		while( i<ed){
			if(Ya[i]>IMX){
				speech_start = i;
				for(j=i;j<ed;j++){
					if(Ya[j]<IMX){
						i++;
						break;
					}
					if(Ya[j]>IMN){
						for(k=j;k<ed;k++){
							if(Ya[k]<IMX){
								speech_end = k - 1;
								if(speech_end - speech_start > 10){
									speech_index[z++]= speech_start;
									speech_index[z++]= speech_end;
								}
								break;
							}else if(k==(ed-1)){
								speech_end = k;
								if(speech_end - speech_start > 10){
									speech_index[z++]= speech_start;
									speech_index[z++]= speech_end;
								}
							}
						}
						i = k;
						break;
					}
				}
			
			}else i++;
			if(j==(ed)) i = j+1;					
		}	
	return z;
}
void four1(float *data, int nn, int isign)
{
    int n, mmax, m, j, istep, i;
    float wtemp, wr, wpr, wpi, wi, theta;
    float tempr, tempi;
    
    n = nn << 1;
    j = 1;
    for (i = 1; i < n; i += 2) {
		if (j > i) {
			tempr = data[j];     data[j] = data[i];     data[i] = tempr;
			tempr = data[j+1]; data[j+1] = data[i+1]; data[i+1] = tempr;
		}
		m = n >> 1;
		while (m >= 2 && j > m) {
			j -= m;
			m >>= 1;
		}
		j += m;
    }
    mmax = 2;
    while (n > mmax) {
		istep = 2*mmax;
		theta = TWOPI/(isign*mmax);
		wtemp = sin(0.5*theta);
		wpr = -2.0*wtemp*wtemp;
		wpi = sin(theta);
		wr = 1.0;
		wi = 0.0;
		for (m = 1; m < mmax; m += 2) {
			for (i = m; i <= n; i += istep) {
				j =i + mmax;
				tempr = wr*data[j]   - wi*data[j+1];
				tempi = wr*data[j+1] + wi*data[j];
				data[j]   = data[i]   - tempr;
				data[j+1] = data[i+1] - tempi;
				data[i] += tempr;
				data[i+1] += tempi;
			}
			wr = (wtemp = wr)*wpr - wi*wpi + wr;
			wi = wi*wpr + wtemp*wpi + wi;
		}
		mmax = istep;
    }
}


/* Function to compute Discrete Cosine Transform */
void mfcc_coeff(struct mfcc *mfccct, struct mfcc *co_eff,float *abso,int st,int ed) {
	
	float cx[(row_length+10)*real_mel_cof];//10 means five live above and five lines below [cx=[c(ww,:); c; c(nf*ww,:)];] matlab
	float t_cx[(row_length+10)*real_mel_cof];
	float vx[row_length+10][real_mel_cof];
	float ax[row_length+2][real_mel_cof];
	float (*vx_ptr)[real_mel_cof];
	float (*ax_ptr)[real_mel_cof];
	int i,j,k; /* Variable declared to act as counters */	
  	float vf[9]={4.000/60, 3.000/60, 2.000/60, 1.000/60,0.000,-1.000/60, -2.000/60, -3.000/60, -4.000/60};
 	float vf_nomi[9]={1.00};
	float af[3]={0.5,0,-0.5};
	float af_nomi[3]={1};
	float t_ax[(row_length+2)*real_mel_cof];
	float t_ax1[(row_length+2)*real_mel_cof];
	
	
    vx_ptr = vx;    
    ax_ptr = ax;

	for ( i=st; i<=ed; i++) { /* For all the frames (EX.70 Frames) */
  			mfccct->data[i][0] = abso[i];//129 is variable "n2" in method mel_freq_spectrum below
  		for (j=1; j<real_mel_cof; j++ ) { /* For "real_mel_cof" the filters, take 13 filter in Number_Of_Filters */
			mfccct->data[i][j] = 0.0;
			/* Compute Cosine Transform of the Signal */
  			for ( k=0; k<Number_Of_Filters; k++){  
  				mfccct->data[i][j] += co_eff->data[i][k]*cos((PI*(k+1./2.)*j)/Number_Of_Filters);
  			}	  				 			
 			mfccct->data[i][j] *= powf(2,0.5*(j+2-j%2)-j/2); 
  		}
 			//mfccct->data[i][1] /= sqrt(2); //only for the first element			
  	}
	/*this below code STRETCH the mfcct matrix to a straight matrix cx[]
  	 * after that cx[] will be filtered and reshape to cd[] 
  	 */
  	 i=st;
  	 for(j=0;j<real_mel_cof;j++){//[matlab]cx=[c(ww,:); c; c(nf*ww,:)]; cx(:)
  	 	 k=0;
	  	 while(k<5){//five lines above
	  	 	cx[i]= mfccct->data[st][j];
	  	 	i++;k++;
	  	 }
	  	 k=st;
	  	 while(k<=ed){
	  	 	cx[i]= mfccct->data[k][j];
	  	 	i++;k++;
	  	 }
	  	 k=0;
	  	 while(k<5){//five line below
	  	 	cx[i]= mfccct->data[ed][j];
	  	 	i++;k++;
	  	 } 
  	 }
  
  	 	//this filter the cx[] and save results to t_cx(temp_cx)
 		filter(9-1,vf_nomi,vf,((ed-st+11)*real_mel_cof)-1,cx,t_cx,st);
 		reshape(t_cx, vx_ptr,  ed+11,  real_mel_cof,st);//ed+11 := st+10+(ed-st+1) 
 		
  		stretch(vx_ptr, t_ax,  st+8, ed+11,  0, real_mel_cof,st);
  		filter(3-1,af_nomi,af,((ed-st+3)*real_mel_cof)-1,t_ax,t_ax1,st);
 		reshape(t_ax1, ax_ptr,  ed+3,  real_mel_cof,st);//ed+3 := ed+11-8 = st+10+(ed-st+1)-8
 		
 		for(i=st;i<=ed;i++){//[matlab]cx=[c(ww,:); c; c(nf*ww,:)]; cx(:)
	  	 	 for(j=real_mel_cof;j<2*real_mel_cof;j++){
		  	 	mfccct->data[i][j] = vx[i+9][j-real_mel_cof];		  	 	
		  	 }
		  	 for(j=2*real_mel_cof;j<3*real_mel_cof;j++){
		  	 	mfccct->data[i][j] = ax[i+2][j-2*real_mel_cof];		  	
		  	 }
  	 	}
  	 
  return;	 
} 
void reshape(float *in, float (*out)[real_mel_cof], int r, int c,int st){
	//this for-loop to RESHAPE the matrix 	
	int i,j,k;
	for(i=0,k=st;i<c;i++){//[matlab]cx=[c(ww,:); c; c(nf*ww,:)]; cx(:)
  	 	 for(j=st;j<r;j++){
	  	 	out[j][i] = in[k];
	  	 	k++;
	  	 }
  	 }
   return;
}
void stretch(float (*in)[real_mel_cof], float *out, int r1,int r2, int c1,int c2,int st){
	//this for-loop to RESHAPE the matrix 	
	int i,j,k;
	for(i=c1,k=st;i<c2;i++){//[matlab]cx=[c(ww,:); c; c(nf*ww,:)]; cx(:)
  	 	 for(j=r1;j<r2;j++){
	  	 	out[k] = in[j][i];
	  	 	k++;
	  	 }
  	 }
   return;
}


/*
 * This method create a mel filter bank "mel"
 * calculate power spectrum
 * multiply with signal frames to create mfcc_coeff matric
 */
void mel_freq_spectrum(struct buffer *input_data, struct mfcc *mfcc_coeff,float *abso,int st, int ed) {
	float temp,maxi,lr;
	int k,j,i;//,o,ee;
	float pf[127];//127 = n*f0*(exp((p+1)*lr) - 1) - 1
	int fp[127];
	float pm[127];	
	int b[4];
	int n = column_length;
	int p = Number_Of_Filters;
	//int fs = 16000;
	float mel[Number_Of_Filters][half_of_column];//DO NOT USE mel[][]={0.0} -> It cause error and I don't know why >"<	
	//melfb(mel[20][129],20, n, fs);//20 means the number of filters
	float f0 = 700.0000/ Us_fs;	
	lr = (log(1 + 0.500000/f0)) / (p+1);
	/////////////////////////Caculate Mel array///////////////////////////////////////////////////
	for(i=0;i<Number_Of_Filters;i++)
		for(j=0;j<half_of_column;j++)	
			mel[i][j]=0.0;
	
	b[0] = n*f0*(exp(0*lr) - 1) ;//0
	b[1] = n*f0*(exp(lr) - 1)  ;//1
	b[2] = n*f0*(exp(p*lr) - 1)-1;//111
	b[3] = n*f0*(exp((p+1)*lr) - 1) - 1;//126

	for(k = b[0];k<=b[3];k++){
		pf[k] = (log(1 + (k+1.000000)/n/f0)) / lr;
		fp[k] = floor(pf[k]);
		pm[k] = pf[k] - fp[k];
	}
	//r = [fp(b2:b4) 1+fp(1:b3)];
	//c = [b2:b4 1:b3] + 1;
	//v = 2 * [1-pm(b2:b4) pm(1:b3)];
	k=b[1];
	/* mel is an sparse matrix */
	while(k<=b[3]){
		mel[fp[k]-1][k+1]=2.0*(1-pm[k]);
		k++;		
	}
	k=0;
	while(k<=b[2]){
		mel[fp[k]][k+1]=2.0*pm[k];
		k++;
	}
	//////////////////////////END - Caculate Mel array///////////////////////////////////////////////
	/*mulipfy mel array with input signal
	 * find the maximum value and calculate naturan log and save to coeef array
	 */	
	maxi=0.00;
	for(j=st;j<=ed;j++){//for each frame
		temp = 0.0;
		//the right below (For-loop and If-command) used to calculate "log(sum(pw))" where pw is "abs(frame(1:n2,:)).^2;"
		for ( k=0; k<half_of_column ; k++){			 
			input_data->data[j][k].real =((input_data->data[j][k].real)*(input_data->data[j][k].real))+ ((input_data->data[j][k].imag)*(input_data->data[j][k].imag));
			if(input_data->data[j][k].real>maxi) maxi = input_data->data[j][k].real; 
			temp += input_data->data[j][k].real;
		}		
		if(temp!=0)	abso[j]=log(temp);//log(sum(pw)) and saving to "abso" array;used in case c=[log(sum(pw)).' c];
		//this For-loop multiply the "pw" and "mel" matrix to calculate the mfcc_coeff
		for ( i=0; i<p; i++ ){//the number of mel array row
			temp = 0.0;			
			for ( k=0; k<half_of_column ; k++) {
				temp += input_data->data[j][k].real*(mel[i][k]);
			}
			mfcc_coeff->data[j][i]=temp;
		}
	}
	maxi*=0.000001;//this "maxi" value to make coefficient smaller than maxi maxi
	for(j=st;j<=ed;j++){
		for ( i=0; i<p ; i++)
			mfcc_coeff->data[j][i] = (float) log(1.0000*max(mfcc_coeff->data[j][i],maxi));//natural logarith at the end		
	}
	
	
		
	return; /*Return back to Main Function */	
}
/* Function to play back the speech signal */
int min(int a,int b){
	if (a<b)return a;
	else return b; 
}
float max(float a,float b){
	if (a>b & a!=0)return a;/* there is larger than zero because we will calculate log(result)*/
	else if(b!=0) return b;
	else return 1; 
}
/*trow_length is the number of row of x
 * y is code - vector quantization
 * out put is observation that is input to calculate loglik
 */
void disteu( struct mfcc *mfcc_coeff,float (*y)[leng_real_store],int *out,int st,int ed){
	int r,c,t,idx;
	float sum,mini;
	for(r=st;r<=ed;r++){
		mini = 99999999.999999;		
		for(t=0;t<nSymbols;t++){
			sum = 0.0;
			for(c=0;c<leng_real_store;c++){
				sum += (mfcc_coeff->data[r][c] - y[t][c])*(mfcc_coeff->data[r][c] - y[t][c]);
			}
			if(sum<mini){
				 mini = sum;
				 idx  = t; 
			}
		}
		out[r-st] = idx;
	}
}
double run_hmm_fo(struct Hmm *in_hmm,struct Obs *in_obs){
    int j;
    int k;
    int t;
    double log_lik;
   
    int nstates = nStates;
    int length = in_obs->length;
    double alpha[nStates * row_length];
    
   
    /* Initialize alpha variables and accumulate scaling factor */ 	
    for (j = 0; j < nstates; j++) {
        alpha[j] = in_hmm->pi[j] * in_hmm->b[((in_obs->data[0]) * nstates) + j];
    }
    
    
    /* Induction step - calculate alpha variables and apply scaling factor */
    for (t = 1; t < length; t++) {
//        scale[t] = 0;
        for (j = 0; j < nstates; j++) {
            alpha[(t * nstates) + j] = 0;
            for (k = 0; k < nstates; k++) {
                alpha[(t * nstates) + j] += alpha[((t-1) * nstates) + k] * 
                                            in_hmm->a[(k * nstates) + j] *
                                            in_hmm->b[((in_obs->data[t]) * nstates) + j];				
            }
        }
    }
    
    /* Calculate the log10(likelihood) */
    log_lik = 0;

    for (t = nStates * length-1; t >= nStates * length - nstates; t--) {
    	log_lik += alpha[t];
    }
    return  log(log_lik);
}


short playback() {
	

	column++; /* Variable to store the index of speech sample in a frame */
	if ( column >= column_length ) { /* If Colum >=256 reset it to zero
								  * and increment the frame number */
		column = 0;		/* initialize the sample number back to zero */
		row++; 	/* Increment the Frame Number */
	}
	if ( row >= row_length ) { /* If Total Frame Number reaches 100 initialize
							* row to be zero
							* and change the program control inidcating
							* end of playback */
		program_control = 2; /* End of Playback */
		row = 0; /* Initialize the frame number back to zero */
	}
	return ((int)real_buffer.data[row][column].real); /* Return the stored speech Sample */
}
void record_sound(void){
  short recording = 0;
  short playing = 0;  
  	
      reidx=0;
      recording = 1;
      while (recording == 1)
      {

      }
  
      reidx=0;
      playing = 1;
      while (playing == 1){
        DSK6713_LED_on(3);             	//turn on LED#0
        output_left_sample(record[reidx++]); 		//input data
  	    if (reidx>5000000){
  	        playing = 0;
  	        DSK6713_LED_off(3);
  	    }
      } 
     
}

//float run_hmm_bwa( struct Hmm *hmm, 
//                   struct Obs *in_obs, 
//                    int iterations, 
//                    float threshold) 
//{
//
//    int iter;
//    float new_log_lik;
//    float old_log_lik = 0;
//    float *alpha;
//    float *scale;
//    float *beta;
//    float *gamma_sum;
//    float *xi_sum;
//   // float *a = hmm->a;
//   // float *b = hmm->b;
//    //float *pi = hmm->pi;
//    int nstates = nStates;
//    int nsymbols = nSymbols;
//    //int *obs = in_obs->data;
//    int length = in_obs->length;
//    int j;
//    int k;
//
//    alpha = (float *) malloc(sizeof(float) * nstates * length);
//    scale = (float *) malloc(sizeof(float) * length);
//    beta = (float *) malloc(sizeof(float) * nstates * length);
//    gamma_sum = (float *) malloc(sizeof(float) * nstates);
//    xi_sum = (float *) malloc(sizeof(float) * nstates * nstates);
//    
//    /* Run BWA for either max iterations or until threshold is reached */
//    for (iter = 0; iter < iterations; iter++) {
//    
//        new_log_lik = calc_alpha(hmm, in_obs, alpha, scale);
//        calc_beta(hmm, in_obs, scale, beta);
//        calc_gamma_sum(alpha, beta, nstates, length, gamma_sum);
//        calc_xi_sum(hmm, in_obs, alpha, beta, xi_sum);
//        estimate_a(alpha, beta, nstates, length, gamma_sum, xi_sum, hmm->a);
//        estimate_b(hmm, in_obs, alpha, beta, gamma_sum);
//        estimate_pi(alpha, beta, nstates, hmm->pi);
//
//        /* check log_lik vs. threshold */
//        if (threshold > 0 && iter > 0) {
//            if (fabs(pow(10,new_log_lik) - pow(10,old_log_lik)) < threshold) {
//                break;
//            }
//        }
//            
//        old_log_lik = new_log_lik;   
//
//    }
//    
//    /* Free memory */
//    free(alpha);
//    free(scale);
//    free(beta);
//    free(gamma_sum);
//    free(xi_sum);
//    
//    return new_log_lik;
//}
//    
///* Calculates the forward variables (alpha) for an HMM and obs. sequence */
//float calc_alpha(struct Hmm *in_hmm,struct Obs *in_obs, float *alpha, float *scale) 
//{
//
//    int j;
//    int k;
//    int t;
//    float log_lik;
//   // float *a = in_hmm->a;
//   // float *b = in_hmm->b;
//   // float *pi = in_hmm->pi;
//    int nstates = nStates;
//    //int *obs = in_obs->data;
//    int length = in_obs->length;
//    
//    /* Initialize alpha variables and accumulate scaling factor */
//    scale[0] = 0;
//    for (j = 0; j < nstates; j++) {
//        alpha[j] = in_hmm->pi[j] * in_hmm->b[(in_obs->data[0] * nstates) + j];
//        scale[0] += alpha[j];
//    }
//    
//    /* Scale initial variables */
//    for (j = 0; j < nstates; j++) {
//        alpha[j] /= scale[0];
//    }
//    
//    /* Induction step - calculate alpha variables and apply scaling factor */
//    for (t = 1; t < length; t++) {
//        scale[t] = 0;
//        for (j = 0; j < nstates; j++) {
//            alpha[(t * nstates) + j] = 0;
//            for (k = 0; k < nstates; k++) {
//                alpha[(t * nstates) + j] += alpha[((t-1) * nstates) + k] * 
//                                            in_hmm->a[(k * nstates) + j] *
//                                            in_hmm->b[(in_obs->data[t] * nstates) + j];
//            }
//            scale[t] += alpha[(t * nstates) + j];
//        }
//        for (j = 0; j < nstates; j++) {
//            alpha[(t * nstates) + j] /= scale[t];
//        }
//    }
//    
//    /* Calculate the log10(likelihood) */
//    log_lik = 0;
//    for (t = 0; t < length; t++) {
//        log_lik += log10(scale[t]);
//    }
//
//    return log_lik;
//}
//
///* Calculates the backwards variables (beta) */
//void calc_beta(struct Hmm *in_hmm, struct Obs *in_obs, float *scale, float *beta)
//{
//
//    int i;
//    int j;
//    int t;
//   // float *a = in_hmm->a;
//    //float *b = in_hmm->b;
//    int nstates = nStates;
//   // int *obs = in_obs->data;
//    int length = in_obs->length;
//    
//    /* Initialize beta variables with scaling factor*/
//    for (j = 0; j < nstates; j++) {
//        beta[((length - 1) * nstates) + j] = 1/scale[length - 1];
//    }
//    
//    /* Induction step - calculate beta variables usign scaling factor */
//    for (t = length - 2; t >= 0; t--) {
//        for (i = 0; i < nstates; i++) {
//            beta[(t * nstates) + i] = 0;
//            for (j = 0; j < nstates; j++) {
//                beta[(t * nstates) + i] += beta[((t+1) * nstates) + j] * 
//                                            in_hmm->a[(i * nstates) + j] * 
//                                            in_hmm->b[(in_obs->data[t+1] * nstates) + j] /
//                                            scale[t];
//                
//            }
//        }
//    }
//}
//
///* Calculate the sum of the gamma variables */
//void calc_gamma_sum(    float *alpha, 
//                        float *beta, 
//                        int nstates, 
//                        int length, 
//                        float *gamma_sum)
//{
//
//    int j;
//    int t;
//    
//    /* Sum gamma values by multiplying alpha by beta values */
//    for (j = 0; j < nstates; j++) {
//        gamma_sum[j] = 0;
//        for (t = 0; t < length; t++) {
//            gamma_sum[j] += alpha[(t * nstates) + j] * beta[(t * nstates) + j];
//        }
//    }
//}
//
///* Calculate the sum of the xi variables */
//void calc_xi_sum(   struct Hmm *in_hmm, 
//                    struct Obs *in_obs, 
//                    float *alpha, 
//                    float *beta, 
//                    float *xi_sum)
//{
//
//   // float *a = in_hmm->a;
//  //  float *b = in_hmm->b;
//    int nstates = nStates;
//    //int *obs = in_obs->data;
//    int length = in_obs->length;
//    float pr_ab;
//    float *xi;
//    int t;
//    int i;
//    int j;
//    
//    xi = (float *) malloc(sizeof(float) * nstates * nstates);
//    
//    /* Initialize xi sum */
//    for (i = 0; i < nstates; i++) {
//        for (j = 0; j < nstates; j++) {
//            xi_sum[(i * nstates) + j] = 0;
//        }
//    }
//    
//    /* Sum xi values */
//    for (t = 0; t < length - 1; t++) {
//    
//        /* Pr[alpha*beta] = sum(alpha(t,:).*beta(t,:)) */
//        pr_ab = 0;
//        for (i = 0; i < nstates; i++) {
//            pr_ab += alpha[(t * nstates) + i] * beta[(t * nstates) + i];
//        }
//        
//        /* xi[i,j] = alpha[i]*A[i,j]*B[O[t],j]*beta[j]/Pr[alpha*beta] */
//        for (i = 0; i < nstates; i++) {
//            for (j = 0; j < nstates; j++) {
//                xi[(i  *nstates) + j] = alpha[(t * nstates) + i] *
//                                        in_hmm->a[(i * nstates) + j] * 
//                                        in_hmm->b[(in_obs->data[t+1] * nstates) + j] *
//                                        beta[((t+1) * nstates) + j] /
//                                        pr_ab;
//            }
//        }
//        
//        /* xi_sum[i,j] += xi[i,j] */
//        for (i = 0; i < nstates; i++) {
//            for (j = 0; j < nstates; j++) {
//                xi_sum[(i * nstates) + j] += xi[(i * nstates) + j];
//            }
//        }
//    }
//                                        
//    free(xi);
//}
//
///* Re-estimate the state transition probabilities */
//void estimate_a(    float *alpha, 
//                    float *beta, 
//                    int nstates,
//                    int length, 
//                    float *gamma_sum, 
//                    float *xi_sum, 
//                    float *a)
//{
//
//    float sum_ab = 0;
//    float sum_a;
//    int i;
//    int j;
//    
//    for (i = 0; i < nstates; i++) {
//        sum_ab += alpha[((length-1) * nstates) + i] * 
//                    beta[((length-1) * nstates) + i];
//    }
//    
//    for (i = 0; i < nstates; i++) {
//        for (j = 0; j < nstates; j++) {
//            a[(i * nstates) + j] = xi_sum[(i * nstates) + j] / (gamma_sum[i] -
//                                (alpha[((length-1) * nstates) + i] * 
//                                beta[((length-1) * nstates) + i] / sum_ab));
//        }
//        sum_a = 0;
//        for (j = 0; j < nstates; j++) {
//            sum_a += a[(i * nstates) + j];
//        }
//        for (j = 0; j < nstates; j++) {
//            a[(i * nstates) + j] = a[(i * nstates) + j] / sum_a;
//        }
//    }
//}
//
///* Re-estimate the symbol output probabilities */
//void estimate_b(    struct Hmm *in_hmm, 
//                    struct Obs *in_obs, 
//                    float *alpha, 
//                    float *beta, 
//                    float *gamma_sum)
//{
//
//   // float *a = in_hmm->a;
//   // float *b = in_hmm->b;
//    int nstates = nStates;
//    int nsymbols = nSymbols;
//   // int *obs = in_obs->data;
//    int length = in_obs->length;
//    float *out_sum;
//    float sum_ab;
//    float sum_b;
//    int i;
//    int k;
//    int t;
//    
//    out_sum = (float *) malloc(sizeof(float) * nstates * nsymbols);
//    
//    /* Initialize sum */
//    for (k = 0; k < nsymbols; k++) {
//        for (i = 0; i < nstates; i++) {
//            out_sum[(k * nstates) + i] = 0;
//        }
//    }
//    
//    /* Calculate B values */
//    for (k = 0; k < nsymbols; k++) {
//        for (t = 0; t < length; t++) {
//            if (in_obs->data[t] == k) {
//                sum_ab = 0;
//                for (i = 0; i < nstates; i++) {
//                    sum_ab += alpha[(t * nstates) + i] * 
//                                beta[(t * nstates) + i];
//                }
//                for (i = 0; i < nstates; i++) {
//                    out_sum[(k * nstates) + i] += alpha[(t * nstates) + i] * 
//                                                    beta[(t * nstates) + i] / 
//                                                    sum_ab;
//                }
//            }     
//        }
//        for (i = 0; i < nstates; i++) {
//            in_hmm->b[(k * nstates) + i] = out_sum[(k * nstates) + i] / gamma_sum[i];
//        }
//    }
//    
//    /* Normalize B values and ensure no B values equal 0 */
//    for (i = 0; i < nstates; i++) {
//        sum_b = 0;
//        for (k = 0; k < nsymbols; k++) {
//            sum_b += in_hmm->b[(k * nstates) + i];
//        }
//        for (k = 0; k < nsymbols; k++) {
//            if (in_hmm->b[(k * nstates) + i] == 0) {
//                in_hmm->b[(k * nstates) + i] = 1e-10;
//            } else {
//                in_hmm->b[(k * nstates) + i] = in_hmm->b[(k * nstates) + i] / sum_b;
//            }
//        }
//    }
//    
//            
//    free(out_sum);
//}
//
///* Re-estimate the initial state probabilities */
//void estimate_pi(float *alpha, float *beta, int nstates, float *pi)
//{
//    float sum_ab = 0;
//    int i;
//    
//    /* Sum initial alpha and beta values */
//    for (i = 0; i < nstates; i++) {
//        sum_ab += alpha[i] * beta[i];
//    }
//    
//    /* Calculate initial state probabilities */
//    for (i = 0; i < nstates; i++) {
//        pi[i] = (alpha[i] * beta[i]) / sum_ab;
//    }
//}

