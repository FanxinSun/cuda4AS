#pragma once

#include <stddef.h>
#include <stdint.h>

#ifndef __align__
#define __align__(n) __attribute__((aligned(n)))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct char1 { signed char x; } char1;
typedef struct __align__(2) char2 { signed char x, y; } char2;
typedef struct char3 { signed char x, y, z; } char3;
typedef struct __align__(4) char4 { signed char x, y, z, w; } char4;

typedef struct uchar1 { unsigned char x; } uchar1;
typedef struct __align__(2) uchar2 { unsigned char x, y; } uchar2;
typedef struct uchar3 { unsigned char x, y, z; } uchar3;
typedef struct __align__(4) uchar4 { unsigned char x, y, z, w; } uchar4;

typedef struct short1 { short x; } short1;
typedef struct __align__(4) short2 { short x, y; } short2;
typedef struct short3 { short x, y, z; } short3;
typedef struct __align__(8) short4 { short x, y, z, w; } short4;

typedef struct ushort1 { unsigned short x; } ushort1;
typedef struct __align__(4) ushort2 { unsigned short x, y; } ushort2;
typedef struct ushort3 { unsigned short x, y, z; } ushort3;
typedef struct __align__(8) ushort4 { unsigned short x, y, z, w; } ushort4;

typedef struct int1 { int x; } int1;
typedef struct __align__(8) int2 { int x, y; } int2;
typedef struct int3 { int x, y, z; } int3;
typedef struct __align__(16) int4 { int x, y, z, w; } int4;

typedef struct uint1 { unsigned int x; } uint1;
typedef struct __align__(8) uint2 { unsigned int x, y; } uint2;
typedef struct uint3 { unsigned int x, y, z; } uint3;
typedef struct __align__(16) uint4 { unsigned int x, y, z, w; } uint4;

typedef struct long1 { long x; } long1;
typedef struct __align__(16) long2 { long x, y; } long2;
typedef struct long3 { long x, y, z; } long3;
typedef struct __align__(16) long4 { long x, y, z, w; } long4;

typedef struct ulong1 { unsigned long x; } ulong1;
typedef struct __align__(16) ulong2 { unsigned long x, y; } ulong2;
typedef struct ulong3 { unsigned long x, y, z; } ulong3;
typedef struct __align__(16) ulong4 { unsigned long x, y, z, w; } ulong4;

typedef struct longlong1 { long long x; } longlong1;
typedef struct __align__(16) longlong2 { long long x, y; } longlong2;
typedef struct longlong3 { long long x, y, z; } longlong3;
typedef struct __align__(16) longlong4 { long long x, y, z, w; } longlong4;

typedef struct ulonglong1 { unsigned long long x; } ulonglong1;
typedef struct __align__(16) ulonglong2 { unsigned long long x, y; } ulonglong2;
typedef struct ulonglong3 { unsigned long long x, y, z; } ulonglong3;
typedef struct __align__(16) ulonglong4 { unsigned long long x, y, z, w; } ulonglong4;

typedef struct float1 { float x; } float1;
typedef struct __align__(8) float2 { float x, y; } float2;
typedef struct float3 { float x, y, z; } float3;
typedef struct __align__(16) float4 { float x, y, z, w; } float4;

typedef struct double1 { double x; } double1;
typedef struct __align__(16) double2 { double x, y; } double2;
typedef struct double3 { double x, y, z; } double3;
typedef struct __align__(16) double4 { double x, y, z, w; } double4;
typedef struct __align__(16) double4_16a { double x, y, z, w; } double4_16a;
typedef struct __align__(32) double4_32a { double x, y, z, w; } double4_32a;
typedef struct __align__(16) longlong4_16a { long long x, y, z, w; } longlong4_16a;
typedef struct __align__(32) longlong4_32a { long long x, y, z, w; } longlong4_32a;
typedef struct __align__(16) ulonglong4_16a { unsigned long long x, y, z, w; } ulonglong4_16a;
typedef struct __align__(32) ulonglong4_32a { unsigned long long x, y, z, w; } ulonglong4_32a;

#ifdef __cplusplus
}
#endif
