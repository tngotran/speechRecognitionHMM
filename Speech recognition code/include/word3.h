#define nStates 6
#define nSymbols 128
#define nA 	nStates*nStates
#define nB  nStates*nSymbols

  
  float a3[nA] =  {  0.0021,    0.9979,         0,         0,         0,         0
,         0,    0.0512,    0.9488,         0,         0,         0
,         0,         0,    0.0842,    0.9158,         0,         0
,         0,         0,         0,    0.8729,    0.1271,         0
,         0,         0,         0,         0,    0.0002,    0.9998
,         0,         0,         0,        0,         0,    1.0000};
		 
  float b3[nB]={ 0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0147,    0.0148,    0.0145,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0075,    0.0873
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0147,    0.0073,    0.0073,    0.0134,    0.0174,    0.0100
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.1666,    0.0088,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0221,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0079,    0.0160,    0.0091,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0107,    0.0105,    0.0074
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0077,    0.0364,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0096,    0.0141,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0265,    0.0089,    0.1055
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0105,    0.0074,    0.0049
,    0.0074,    0.0146,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0146,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0078,    0.0158,    0.0094,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0053,    0.0075,    0.1455
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0074,    0.0153,    0.0552,    0.0098,    0.0050
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0147,    0.0152,    0.0140,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0147,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0109,    0.0202,    0.0056
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0092,    0.0135,    0.0068
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0147,    0.0147,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0061,    0.0272,    0.0492
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0151,    0.0082,    0.0049
,    0.0148,    0.0307,    0.0280,    0.0103,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0089,    0.0093,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
,    0.0074,    0.0073,    0.0073,    0.0052,    0.0074,    0.0049
};

