#ifndef _RESNET_CPP_RESNET_EXPORT_H_
#define _RESNET_CPP_RESNET_EXPORT_H_

#ifdef _MSC_VER

#ifdef ResNetCPP_EXPORTS
#define RESNET_API __declspec(dllexport)
#else
#define RESNET_API __declspec(dllimport)
#endif // ResNetCPP_EXPORTS

#else
#define RESNET_API
#endif // _MSC_VER

#endif // _RESNET_CPP_RESNET_EXPORT_H_