#include <math.h>
#include <R.h>
#include <R_ext/Applic.h>
#include "Rinternals.h"
#include "R_ext/Rdynload.h"

// use the binary heap data structure of a fixed size K
// for computing Kth smallest element
// implemented using an array
// indices for heap elements start at 1
// the array only keeps the K smallest elements

void swap(double *x, double *y) {
  double temp = *x;
  *x = *y;
  *y = temp;
}

void sink(double *a, int K, int i) {
  while (2*i <= K) {
    int j = 2*i;
    if (j<K && a[j]<a[j+1]) j++;
    if (a[i]>=a[j]) break;
    swap(&a[i],&a[j]);
    i = j;
  }
}

void buildMaxHeap(double *a, int K) {
  for(int i=K/2;i>=1;i--) sink(a,K,i);
}

//kth smallest of absolute values
double ksav(double *a, int size, int K) {
  int i;
  double heap[K+1];
  for(i=0;i<K;i++) heap[i+1] = fabs(a[i]);
  buildMaxHeap(heap,K);
  for(i=K;i<size;i++) {
    double abs = fabs(a[i]);
    if(abs<heap[1]) {
      heap[1]=abs;
      sink(heap,K,1);
    }
  }
  return heap[1];
}
