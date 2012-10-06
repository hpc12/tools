#pragma OPENCL EXTENSION cl_khr_fp64: enable

__kernel void sum(
    __global const double *a,
    __global const double *b, 
    __global double *c,
    long n)
{
  int gid = get_global_id(0);
  if (gid < n)
    c[gid] = a[gid] + b[gid];
}
