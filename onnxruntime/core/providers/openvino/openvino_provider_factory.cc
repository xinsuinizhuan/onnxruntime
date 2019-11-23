// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/providers/openvino/openvino_provider_factory.h"
#include "openvino_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {
struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(const char* device, const char* precision) : device_(device),precision_(precision) {
  }
  ~OpenVINOProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  const char* device_;
  const char* precision_;
};

std::unique_ptr<IExecutionProvider> OpenVINOProviderFactory::CreateProvider() {
  OpenVINOExecutionProviderInfo info;
  info.device = device_;
  info.precision = precision_;
  return onnxruntime::make_unique<OpenVINOExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(
    const char* device_id,
    const char* precision) {
  return std::make_shared<onnxruntime::OpenVINOProviderFactory>(device_id,precision);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_OpenVINO,
                    _In_ OrtSessionOptions* options, const char* device_id,
                    const char* precision) {
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_OpenVINO(device_id,precision));
  return nullptr;
}
