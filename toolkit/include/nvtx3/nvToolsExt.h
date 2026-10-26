#ifndef NVTOOLSEXT_H
#define NVTOOLSEXT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NVTX_VERSION 3
#define NVTX_COLOR_UNKNOWN 0
#define NVTX_COLOR_ARGB 1

#define NVTX_MESSAGE_UNKNOWN 0
#define NVTX_MESSAGE_TYPE_ASCII 1
#define NVTX_MESSAGE_TYPE_UNICODE 2
#define NVTX_MESSAGE_TYPE_REGISTERED 3

#define NVTX_PAYLOAD_UNKNOWN 0
#define NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 1
#define NVTX_PAYLOAD_TYPE_INT64 2
#define NVTX_PAYLOAD_TYPE_DOUBLE 3
#define NVTX_PAYLOAD_TYPE_UNSIGNED_INT32 4
#define NVTX_PAYLOAD_TYPE_INT32 5
#define NVTX_PAYLOAD_TYPE_FLOAT 6

typedef struct nvtxDomainRegistration_st* nvtxDomainHandle_t;
typedef struct nvtxStringRegistration_st* nvtxStringHandle_t;

typedef union nvtxMessageValue_u {
    const char* ascii;
    const wchar_t* unicode;
    nvtxStringHandle_t registered;
} nvtxMessageValue_t;

typedef struct nvtxEventAttributes_v2 {
    uint16_t version;
    uint16_t size;
    uint32_t category;
    int32_t colorType;
    uint32_t color;
    int32_t payloadType;
    int32_t reserved0;
    union {
        uint64_t ullValue;
        int64_t llValue;
        double dValue;
        uint32_t uiValue;
        int32_t iValue;
        float fValue;
    } payload;
    int32_t messageType;
    nvtxMessageValue_t message;
} nvtxEventAttributes_t;

#define NVTX_EVENT_ATTRIB_STRUCT_SIZE ((uint16_t)(sizeof(nvtxEventAttributes_t)))

static inline nvtxDomainHandle_t nvtxDomainCreateA(const char* name) {
    (void)name;
    return (nvtxDomainHandle_t)(uintptr_t)1;
}

static inline void nvtxDomainDestroy(nvtxDomainHandle_t domain) {
    (void)domain;
}

static inline nvtxStringHandle_t nvtxDomainRegisterStringA(nvtxDomainHandle_t domain, const char* str) {
    (void)domain;
    (void)str;
    return (nvtxStringHandle_t)(uintptr_t)1;
}

static inline int nvtxDomainRangePushEx(nvtxDomainHandle_t domain, const nvtxEventAttributes_t* attribs) {
    (void)domain;
    (void)attribs;
    return 0;
}

static inline int nvtxDomainRangePop(nvtxDomainHandle_t domain) {
    (void)domain;
    return 0;
}

static inline void nvtxDomainMarkEx(nvtxDomainHandle_t domain, const nvtxEventAttributes_t* attribs) {
    (void)domain;
    (void)attribs;
}

static inline int nvtxRangePushA(const char* message) {
    (void)message;
    return 0;
}

static inline int nvtxRangePop(void) {
    return 0;
}

static inline void nvtxMarkA(const char* message) {
    (void)message;
}

#ifdef __cplusplus
}
#endif

#endif // NVTOOLSEXT_H
