/*****************************************************************************
 *
 *
 * filter_edge.h
 *
 *
 * Include file containing the set of filter edges
 * Eges are precomputed for efficiency.
 *
 *
 *
 *
 * Written by Vasanthan Rangan and Sowmya Narayanan
 *
 *
 *
 *
 *
 *
 *****************************************************************************/

#define Number_Of_Filters 20 /* Total Number of Filters */
 
 float H[Number_Of_Filters+2] = { 
 0.0,2.349535731,4.945514224,7.813784877,10.98290838,
 14.48444125,18.35324982,22.62785770,27.35082918,
 32.56919306,38.33491098,44.70539514,51.74407917,
 59.52105066,68.11374874,77.60773478,88.09754483,
 99.68763091,112.4934010,126.6423682,142.2754203,0.0 };
 

