#define nStates 6
#define nSymbols 128
#define nA 	nStates*nStates
#define nB  nStates*nSymbols

  
  float a2[nA] =  {   0.6841,    0.3159,         0,         0,         0,         0
,         0,    0.8893,    0.1107,         0,         0,         0
,         0,         0,    0.8713,    0.1287,         0,         0
,         0,         0,         0,    0.7129,    0.2871,         0
,        0,         0,         0,         0,    0.8076,    0.1924
,        0,         0,         0,         0,         0,    1.0000
 };
		 
  float b2[nB]={0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0130,    0.0169,    0.0060,    0.0069,    0.0087,    0.0098
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0399,    0.0153,    0.0070,    0.0069,    0.0065,    0.0059
,    0.0260,    0.0057,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0073,    0.0142,    0.0080,    0.0071,    0.0068,    0.0233
,    0.0065,    0.0335,    0.0063,    0.0071,    0.0086,    0.0157
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0082,    0.0196,    0.0088,    0.0065,    0.0059
,    0.0067,    0.0079,    0.0324,    0.0081,    0.0098,    0.0088
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0076,    0.0724,    0.0060,    0.0069,    0.0065,    0.0118
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0084,    0.0095,    0.0062,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0058,    0.0061,    0.0079,    0.0114,    0.0062
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0057,    0.0599,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0070,    0.0702
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0252,    0.0064,    0.0063,    0.0545,    0.0074,    0.0347
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0079,    0.0164
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0069,    0.0159,    0.0067,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0113,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0345,    0.0150,    0.0080,    0.0208,    0.0109,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0084,    0.0125,    0.0143,    0.0138,    0.0073,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0130,    0.0056,    0.0060,    0.0069,    0.0065,    0.0236
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0118
,    0.0070,    0.0832,    0.0071,    0.0069,    0.0066,    0.0117
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0080,    0.0140,    0.0145,    0.0119,    0.0077
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0064,    0.0115,    0.0082,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0234,    0.0088,    0.0100,    0.0081,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0118
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0294
,    0.0065,    0.0058,    0.0062,    0.0088,    0.0206,    0.0088
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0419,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0129,    0.0058,    0.0061,    0.0104,    0.0096,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0061,    0.0126,    0.0122,    0.0068,    0.0059
,    0.0065,    0.0057,    0.0061,    0.0070,    0.0079,    0.0161
,    0.0065,    0.0056,    0.0659,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0196,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0108,    0.0065,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0075,    0.0109
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0098,    0.0141,    0.0060,    0.0074,    0.0968,    0.0063
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0118
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0068,    0.0171,    0.0068,    0.0138,    0.0230,    0.0074
,    0.0065,    0.0057,    0.0061,    0.0070,    0.0093,    0.0091
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0126,    0.0060,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0078,    0.0097,    0.0071,    0.0105,    0.0140
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0057,    0.0083,    0.0211,    0.0099,    0.0060
,    0.0065,    0.0056,    0.0060,    0.0069,    0.0065,    0.0059
,    0.0065,    0.0057,    0.0060,    0.0069,    0.0081,    0.0102   };
