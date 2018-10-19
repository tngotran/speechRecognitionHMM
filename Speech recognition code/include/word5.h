#define nStates 6
#define nSymbols 128
#define nA 	nStates*nStates
#define nB  nStates*nSymbols

  
  float a5[nA] =  {0.8579,    0.1421,         0,         0,         0,         0
,         0,    0.0621,    0.9379,         0,         0,         0
,         0,         0,    0.0247,    0.9753,         0,         0
,         0,         0,         0,    0.0369,    0.9631,         0
,         0,         0,         0,         0,    0.8288,    0.1712
,        0,         0,         0,         0,         0,    1.0000
 };
		 
  float b5[nB]={0.0299,    0.0108,    0.0083,    0.0076,    0.0063,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0334,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0111,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0167,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0061,    0.0103,    0.0107,    0.0078,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0075,    0.0078,    0.0117,    0.0048
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0075,    0.0310,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0123,    0.0047
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0090,    0.0144,    0.0096,    0.0081,    0.0064,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0144,    0.0064,    0.0185
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0083,    0.0117,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0111,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0121,    0.0049
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0111,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0111,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0099,    0.0082,    0.0081,    0.0076,    0.0114,    0.0145
,    0.0275,    0.0144,    0.0081,    0.0076,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0057,    0.0081,    0.0103,    0.0106,    0.0065,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0138
,    0.0056,    0.0074,    0.0075,    0.0083,    0.0116,    0.0046
,    0.0056,    0.0103,    0.0144,    0.0117,    0.0067,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0159,    0.0083,    0.0076,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0081,    0.0160,    0.0108,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0060,    0.0085,    0.0112,    0.0107,    0.0176,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0223,    0.0075,    0.0077,    0.0080,    0.0776,    0.0065
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0092
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0171,    0.0078,    0.0131,    0.0080,    0.0064,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0069,    0.1055
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0080,    0.0125,    0.0090,    0.0116,    0.0053
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0069,    0.0119,    0.0129,    0.0099,    0.0065,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0994,    0.0080,    0.0077,    0.0080,    0.0292,    0.0058
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0138
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0230
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0075,    0.0077,    0.0088,    0.0109,    0.0047
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0165,    0.0076,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0075,    0.0082,    0.0105,    0.0091,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0143,    0.0146,    0.0077,    0.0071,    0.1378
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0113,    0.0077,    0.0078,    0.0131,    0.0068,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0095,    0.0096,    0.0074,    0.0074,    0.0062,    0.0138
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0276
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0060,    0.0125,    0.0091,    0.0074,    0.0062,    0.0138
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0057,    0.0078,    0.0084,    0.0108,    0.0203,    0.0050
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0105,    0.0081,    0.0075,    0.0074,    0.0062,    0.0691
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0077,    0.0086,    0.0108,    0.0082,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0111,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0089,    0.0099,    0.0078,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0075,    0.0116,    0.0080,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046
,    0.0056,    0.0074,    0.0074,    0.0074,    0.0062,    0.0046

};
