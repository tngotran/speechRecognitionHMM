#define nStates 6
#define nSymbols 128
#define nA 	nStates*nStates
#define nB  nStates*nSymbols

  
  float a6[nA] =  {  0.8592,    0.1408,         0,         0,         0,         0
,         0,    0.2921,    0.7079,         0,         0,         0
,         0,         0,    0.0331,    0.9669,         0,         0
,         0,         0,         0,    0.0549,    0.9451,         0
,         0,         0,         0,         0,    0.6367,    0.3633
,         0,         0,         0,         0,         0,    1.0000
};
		 
  float b6[nB]={  0.0089,    0.0119,    0.0123,    0.0120,    0.0093,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0207,    0.0084,    0.0074,    0.0073,    0.0068,    0.0052
,    0.1234,    0.0084,    0.0074,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0379,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0113,    0.0068
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0310
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0258
,    0.0108,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0064,    0.0146,    0.0123,    0.0081,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0060,    0.0110,    0.0092,    0.0079,    0.0069,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0076,    0.0142,    0.0069,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0076,    0.0097,    0.0110,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0055,    0.0161,    0.0138,    0.0133,    0.0121,    0.0066
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0075,    0.0083,    0.0308,    0.0066
,    0.0054,    0.0073,    0.0088,    0.0222,    0.0201,    0.0557
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0162,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0207
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0414
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0074,    0.0078,    0.0395,    0.0057
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0074,    0.0143,    0.0075,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0112,    0.0198,    0.0084,    0.0075,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0103
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0072,    0.0112,    0.0080,    0.0075,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0108,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0577,    0.0163,    0.0078,    0.0074,    0.0068,    0.0052
,    0.0117,    0.0195,    0.0083,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0073,    0.0152,    0.0084,    0.0117,    0.0054
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0216,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0097,    0.0081
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0621
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0465
,    0.0055,    0.0110,    0.0141,    0.0145,    0.0105,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0155,    0.0081,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0056,    0.0086,    0.0151,    0.0123,    0.0070,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0107,    0.0103,    0.0138,    0.0118,    0.0074,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0074,    0.0082,    0.0332,    0.0567
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0162,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0103
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0259
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0114,    0.0128,    0.0080,    0.0074,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0073,    0.0078,    0.0127,    0.0147,    0.0053
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052
,    0.0054,    0.0072,    0.0073,    0.0073,    0.0068,    0.0052};
	