// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <stddef.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/compute_capability.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/util/protobuf_parsing_utils.h"
#include "core/common/logging/logging.h"

#include "openvino_execution_provider.h"
#include "core/graph/model.h"
#include "openvino_graph.h"
#include <inference_engine.hpp>

namespace onnxruntime {

constexpr const char* OpenVINO = "OpenVINO";

OpenVINOExecutionProvider::OpenVINOExecutionProvider(OpenVINOExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kOpenVINOExecutionProvider} {
  //ORT_UNUSED_PARAMETER(info);
  requested_device_ = info.device;
  requested_precision_ = info.precision;

  DeviceAllocatorRegistrationInfo device_info({OrtMemTypeDefault, [](int) { return onnxruntime::make_unique<CPUAllocator>(onnxruntime::make_unique<OrtMemoryInfo>(OPENVINO, OrtDeviceAllocator)); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(CreateAllocator(device_info));
}

static ONNX_NAMESPACE::ModelProto GetModelProtoFromFusedNode(const onnxruntime::GraphViewer& graph_viewer) {
  ONNX_NAMESPACE::ModelProto model_proto;
  auto graph_proto = model_proto.mutable_graph();

  for (const auto& node : graph_viewer.Nodes()) {
    node.ToProto(*(graph_proto->add_node()));
  }

  for (const auto& input : graph_viewer.GetInputs()) {
    auto valueInfoProto = graph_proto->add_input();
    *valueInfoProto = input->ToProto();
  }

  for (const auto& output : graph_viewer.GetOutputs()) {
    auto valueInfoProto = graph_proto->add_output();
    *valueInfoProto = output->ToProto();
  }

  for (const auto& initializer : graph_viewer.GetAllInitializedTensors()) {
    graph_proto->add_initializer()->CopyFrom(*initializer.second);
  }

  auto opset = model_proto.add_opset_import();
  opset->set_domain(kOnnxDomain);
  opset->set_version(graph_viewer.DomainToVersionMap().at(kOnnxDomain));
  model_proto.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  return model_proto;
}

//Gets the input count of given node
int GetInputCount(const Node* node, const InitializedTensorSet& initializer_set) {
  int count = 0;
  for (const auto& input : node->InputDefs()) {
    auto name = input->Name();
    auto it = initializer_set.find(name);
    if (it == initializer_set.end()) {
      count++;
    }
  }
  return count;
}

//Checks whether the dimensions of a given node are supported in OpenVINO
bool IsDimensionSupported(const Node* node, std::string dev_id) {
  auto node_inputs = node->InputDefs();
  size_t input_dims = 0;
  if (node_inputs[0]->Shape() != nullptr) {
    input_dims = node_inputs[0]->Shape()->dim_size();
  }

  if (node->OpType().find("Pool") != std::string::npos) {
    if (dev_id == "MYRIAD" || dev_id == "HDDL") {
      if (input_dims != 3 && input_dims != 4)
        return false;
    } else if (input_dims != 4 && input_dims != 5) {
      return false;
    }
  }

  //Only support 4D and 5D Transposes
  if (node->OpType() == "Transpose") {
    if (input_dims == 2 || input_dims == 3 || input_dims > 5)
      return false;
  }

  if (node->OpType() == "Unsqueeze") {
    auto attributes = node->GetAttributes();
    auto axes = attributes["axes"].ints();
    if (input_dims + axes.size() > 5)
      return false;
    if (dev_id == "MYRIAD" || dev_id == "HDDL") {
      if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim(0).dim_value() != 1)
        return false;
    }
  }

  if (node->OpType() == "Reshape") {
    //Don't support Reshape without output dims
    auto node_outputs = node->OutputDefs();
    if (node_outputs[0]->Shape() != nullptr && node_outputs[0]->Shape()->dim_size() == 0)
      return false;

    if (dev_id == "MYRIAD" || dev_id == "HDDL") {
      if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim(0).dim_value() != 1)
        return false;
    }
  }

  if (node->OpType() == "Softmax") {
    //First dimension of Softmax input has to be 1
    if (input_dims != 0) {
      if (node_inputs[0]->Shape()->dim(0).dim_value() != 1)
        return false;
    }

    //3D input not supported on GPU, MYRIAD and HDDL
    if (dev_id == "GPU" || dev_id == "MYRIAD" || dev_id == "HDDL") {
      if (input_dims == 3)
        return false;
    }
  }
  //Only 2D MatMul is supported
  if (node->OpType() == "MatMul") {
    for (size_t i = 0; i < node_inputs.size(); i++) {
      if (node_inputs[i]->Shape() != nullptr) {
        if (node_inputs[i]->Shape()->dim_size() != 2)
          return false;
      }
    }
  }

  if (node->OpType() == "Flatten") {
    if (dev_id == "MYRIAD" || dev_id == "HDDL") {
      if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim(0).dim_value() != 1)
        return false;
    }
  }

  return true;
}

//Checks whether the node is supported by OpenVINO
bool IsOpSupported(std::string name) {
  std::set<std::string> supported_ops = {
      "Add",
      "BatchNormalization",
      "Conv",
      "GlobalAveragePool",
      "Relu",
      "Reshape",
      "Flatten",
      "Gemm",
      "MaxPool",
      "AveragePool",
      "Concat",
      "Dropout",
      "LRN",
      "Softmax",
      "Mul",
      "Sum",
      "Transpose",
      "Identity",
      "MatMul",
      "Unsqueeze",
      "ImageScaler",
      "LeakyRelu",
      "GlobalMaxPool",
      "Div",
      "Sub"};

  auto iter = supported_ops.find(name);
  return iter != supported_ops.end();
}

//Checks if the entire graph is supported by OpenVINO EP and message =s eception if any.

bool CheckGraphSupported(const onnxruntime::GraphViewer& graph_viewer, std::string dev_id, std::string& message) {
  const auto& initializers = graph_viewer.GetAllInitializedTensors();

  auto node_indexes = graph_viewer.GetNodesInTopologicalOrder();

  auto model_proto = GetModelProtoFromFusedNode(graph_viewer);

  auto graph_proto = model_proto.mutable_graph();
  int input_dims = 0;
  int output_dims = 0;
  int num_inputs = graph_viewer.GetInputs().size();
  int num_outputs = graph_viewer.GetOutputs().size();

  //GPU Plugin does not support 1D and 5D input
  if (dev_id == "GPU") {
    for (int i = 0; i < num_inputs; i++) {
      input_dims = graph_proto->input(i).type().tensor_type().shape().dim_size();

      if (input_dims == 1 || input_dims == 5) {
        message = "GPU plugin doesn't support 1D and 5D input";
        return false;
      }
    }
  }

  //GPU Plugin does not support 5D output
  if (dev_id == "GPU") {
    for (int i = 0; i < num_outputs; i++) {
      output_dims = graph_proto->output(i).type().tensor_type().shape().dim_size();

      if (output_dims == 5) {
        message = "GPU plugin doesn't support  5D output";
        return false;
      }
    }
  }

  for (auto index : node_indexes) {
    const auto node = graph_viewer.GetNode(index);

    //Check if the Operation is Supported by OpenVINO
    if (!IsOpSupported(node->OpType())) {
      {
        message = "Operation is not supported by OpenVINO";
        return false;
      }
    }

    auto node_inputs = node->InputDefs();

    //Zero dimension check
    for (size_t i = 0; i < node_inputs.size(); i++) {
      auto name = node_inputs[i]->Name();
      auto it = initializers.find(name);
      if (it == initializers.end() && node_inputs[i]->Shape() != nullptr) {
        if (node_inputs[i]->Shape()->dim_size() == 0) {
          message = "Node_input is zero dimension";
          return false;
        }
      }
    }

    //BatchNormalization cannot take more than 1 input
    if (node->OpType() == "BatchNormalization") {
      if (GetInputCount(node, initializers) > 1) {
        message = "BatchNormalization: Cannot take more than 1 input";
        return false;
      }
    }

    //Conv cannot take more than 1 input
    if (node->OpType() == "Conv") {
      if (GetInputCount(node, initializers) > 1) {
        message = "Conv: Cannot take more than 1 input";
        return false;
      }
    }

    //Reshape should have shape as initializer
    if (node->OpType() == "Reshape") {
      int input_count = GetInputCount(node, initializers);

      if (input_count > 1) {
        message = "Reshape: Shape should be an initializer";
        return false;
      }

      //Myriad and HDDL plugins do not support Reshape with two initializers
      if (dev_id == "MYRIAD" || dev_id == "HDDL")
        if (input_count == 0) {
          message = "Myriad and HDDL plugins do not support Reshape with two initializers ";
          return false;
        }

      if (!IsDimensionSupported(node, dev_id)) {
        message = "Reshape: Dimensions are not supported";
        return false;
      }
    }

    if (node->OpType() == "Flatten") {
      if (!IsDimensionSupported(node, dev_id)) {
        message = "Flatten: Dimensions are not supported";
        return false;
      }

      //Only default axis is supported for MYRIAD and HDDL plugins
      auto attributes = node->GetAttributes();
      auto axis = attributes["axis"].i();
      if (dev_id == "MYRIAD" || dev_id == "HDDL") {
        if (axis != 1) {
          message = "Only default axis is supported for MYRIAD and HDDL plugins";
          return false;
        }
      }
    }

    if (node->OpType() == "Mul" || node->OpType() == "Add" || node->OpType() == "Div" || node->OpType() == "Sub") {
      for (size_t i = 0; i < node->InputDefs().size(); i++) {
        if (node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64) {
          message = "int64 inputs not supported";
          return false;
        }
      }
    }

    if (node->OpType() == "Div") {
      for (size_t i = 0; i < node->InputDefs().size(); i++) {
        if (node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32) {
          message = "int32 inputs not supported for Div";
          return false;
        }
      }
    }

    //MatMul is only supported if it is followed by Add
    if (node->OpType() == "MatMul") {
      for (size_t i = 0; i < node->InputDefs().size(); i++) {
        if (node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
          message = "Input data type should be float";
          return false;
        }
      }

      auto iter = node->OutputNodesBegin();

      if (iter == node->OutputNodesEnd()) {
        message = "MatMul should be followed by Add";
        return false;
      }

      for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
        const auto out_node = graph_viewer.GetNode((*it).Index());

        if (out_node->OpType() != "Add") {
          {
            message = "Matmul should be followed by Add";
            return false;
          }
        }
      }

      if (!IsDimensionSupported(node, dev_id)) {
        message = "Dimension is not supported";
        return false;
      }
    }

    //Dropout , Identity and Concat can't have graph inputs
    if (node->OpType() == "Dropout" || node->OpType() == "Identity" || node->OpType() == "Concat" || node->OpType() == "Gemm") {
      auto graph_inputs = graph_viewer.GetInputs();
      for (const auto& input : node->InputDefs()) {
        auto it = find(graph_inputs.begin(), graph_inputs.end(), input);
        if (it != graph_inputs.end()) {
          {
            message = "Dropout, Identity, Concat, and Gemm can't have graph inputs";
            return false;
          }
        }
      }
    }

    //Attribute auto pad for MaxPool and Average Pool must not be empty or SAME_LOWER
    //Only support 4D and 5D blobs for CPU,GPU
    //Only support 3D and 4D blobs for MYRIAD and HDDL
    if (node->OpType() == "MaxPool" || node->OpType() == "AveragePool") {
      auto attributes = node->GetAttributes();
      auto auto_pad = attributes["auto_pad"].s();
      if (auto_pad == "" || auto_pad == "SAME_LOWER") {
        message = "Attribute auto pad shouldn't be empty or SAME_LOWER for MaxPool and Average Pool";
        return false;
      }

      auto strides_ints = attributes["strides"].ints();
      if (auto_pad == "SAME_UPPER" && strides_ints.size() == 0) {
        message = "Pooling: Generic Error";
        return false;
      }

      //Dilations have to be 1
      auto dilations_ints = attributes["dilations"].ints();
      if (dilations_ints.size() != 0) {
        if (dilations_ints[0] > 1) {
          message = "Pooling: Generic error";
          return false;
        }
      }

      //Don't support ceil_mode = 1
      auto ceil_mode = attributes["ceil_mode"].i();
      if (ceil_mode != 0) {
        message = "Pooling: Ceil mode should be 1";
        return false;
      }

      //Don't support multiple outputs for Pooling
      if (node->OutputDefs().size() > 1) {
        message = "Pooling: Multiple outputs not supported";
        return false;
      }

      if (!IsDimensionSupported(node, dev_id)) {
        message = "Pooling: Dimensions not supported";
        return false;
      }
    }

    //Only support 4D and 5D blobs for CPU,GPU
    //Only support 3D and 4D blobs for MYRIAD and HDDL
    if (node->OpType() == "GlobalMaxPool" || node->OpType() == "GlobalAveragePool") {
      if (!IsDimensionSupported(node, dev_id)) {
        message = "Pooling: Only support 4D and 5D blobs for CPU,GPU, Only support 3D and 4D blobs for MYRIAD and HDDL";
        return false;
      }
    }

    //Transpose with no attr is not supported
    if (node->OpType() == "Transpose") {
      auto attributes = node->GetAttributes();
      auto perm = attributes["perm"].ints();
      if (perm.size() == 0 || perm.size() > 5) {
        message = " Transpose: Tranpose with no attr is not supported. Perm size shouldn't be zero or greater than five";
        return false;
      }

      //String data type is not supported
      const auto* type_proto = node->InputDefs()[0]->TypeAsProto();
      if (type_proto->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_STRING) {
        message = "Transpose: String data type is not supported ";
        return false;
      }

      if (!IsDimensionSupported(node, dev_id)) {
        message = "Transpose: Dimensions are not supported ";
        return false;
      }
    }

    if (node->OpType() == "Unsqueeze") {
      if (!IsDimensionSupported(node, dev_id)) {
        message = "Unsqueeze: Dimensions are not supported ";
        return false;
      }
      const auto* type_proto = node->InputDefs()[0]->TypeAsProto();
      if (type_proto->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) {
        message = "Unsqueeze: Datatype should be float";
        return false;
      }
    }

    //Only support 2D input and axis 1
    if (node->OpType() == "Softmax") {
      if (!IsDimensionSupported(node, dev_id)) {
        message = "Softmax: Dimensions are not supported ";
        return false;
      }

      auto attributes = node->GetAttributes();
      auto axis = attributes["axis"].i();
      if (axis != 1) {
        message = "Softmax: Only default axis is supported";
        return false;
      }
    }

    //Don't support only one input
    if (node->OpType() == "Sum") {
      if (node->InputDefs().size() == 1) {
        message = "Sum: Doesn't support only one input ";
        return false;
      }
    }
  }
  return true;
}

std::string SelectDevice(const onnxruntime::GraphViewer& graph_viewer, std::string requested_device, bool& precision_fp32, std::string& error_msg) {
  InferenceEngine::Core ie;
  std::vector<std::string> available_devices;
  std::vector<std::string> usable_devices;
  available_devices = ie.GetAvailableDevices();
  bool one_device_supports_graph = false;
  bool one_requested_device_found = false;
  bool graph_supported_by_device = false;
  std::string dev_error_msg = "NA";

  std::string requested_devices_not_found = requested_device;
  if(requested_device.size() > 0) {
    //Determine whether requested device(s) are available
    for (auto && dev: available_devices) {
      if(requested_device.find(dev) != std::string::npos) {
        one_requested_device_found = true;
        requested_devices_not_found.erase(requested_devices_not_found.find(dev),dev.size());
      }
    }
    //Strip hetero/multi prefix if present
    if(requested_devices_not_found.find("HETERO:") != std::string::npos) {
      requested_devices_not_found.erase(requested_devices_not_found.find("HETERO:"),7);
    }
    if(requested_devices_not_found.find("MULTI:") != std::string::npos) {
      requested_devices_not_found.erase(requested_devices_not_found.find("MULTI:"),6);
    }
    std::replace(requested_devices_not_found.begin(),requested_devices_not_found.end(), ',', ' ');
    if(requested_devices_not_found.size() > 0 && requested_devices_not_found.find_first_not_of(" ") != std::string::npos) {
      LOGS_DEFAULT(WARNING) << openvino_ep::OpenVINOGraph::log_tag << "Devices (" << requested_devices_not_found.c_str() << ") not found";
    }
    if(!one_requested_device_found) {
      LOGS_DEFAULT(WARNING) << openvino_ep::OpenVINOGraph::log_tag << "No user specified devices found. Defaulting to all available devices";
      requested_device = "";
    }
  }
  
  std::vector<std::string> available_and_requested_devices;
  //Get intersection of available and requested devices (preserve ordering)
  if(requested_device.size() != 0)
  {
    for (auto && dev: available_devices)
      if(requested_device.find(dev) != std::string::npos) {
        available_and_requested_devices.push_back(dev);
        
      }
  }
  else {
    available_and_requested_devices = available_devices;
  }     

  for (auto && dev: available_and_requested_devices) {
    //Check whether the graph is supported on each of the available and requested devices
    graph_supported_by_device = CheckGraphSupported(graph_viewer, dev, dev_error_msg);
    error_msg += dev + ":" + dev_error_msg + ". "; 
    if(graph_supported_by_device) {
      //If one of the devices supports the graph, all of the available devices in heterogenous mode are supporter
      one_device_supports_graph = true;
      if(requested_device.find("HETERO") != std::string::npos) {
        usable_devices = available_and_requested_devices;
        break;        
      }
      usable_devices.push_back(dev);
    }
  }
  if(!one_device_supports_graph) {
    throw error_msg.c_str();
  }

  if(usable_devices.size() == 1)
  {
    return usable_devices[0];
  }
  else {
    //Set precision based on usable devices. If VPU is present, must use FP16. Otherwise, all devices support 32 bit models
    if(std::find(usable_devices.begin(),usable_devices.end(),"HDDL") != usable_devices.end() || 
       std::find(usable_devices.begin(),usable_devices.end(),"MYRIAD") !=usable_devices.end()) {
        precision_fp32 = false;
    }

    std::string allDevices = "MULTI:HDDL,MYRIAD,FPGA,GPU,CPU";

    if(requested_device.find("HETERO") != std::string::npos) {
      allDevices = "HETERO:HDDL,MYRIAD,FPGA,GPU,CPU";
    }

    //prioritize usable devices.
    if(std::find(usable_devices.begin(),usable_devices.end(),"HDDL") == usable_devices.end()) {
      allDevices.erase(allDevices.find("HDDL"),5);
    }
    if(std::find(usable_devices.begin(),usable_devices.end(),"MYRIAD") == usable_devices.end()) {
      allDevices.erase(allDevices.find("MYRIAD"),7);
    }
    if(std::find(usable_devices.begin(),usable_devices.end(),"FPGA") == usable_devices.end()) {
      allDevices.erase(allDevices.find("FPGA"),5);
    }
    if(std::find(usable_devices.begin(),usable_devices.end(),"GPU") == usable_devices.end()) {
      allDevices.erase(allDevices.find("GPU"),4);
    }
    if(std::find(usable_devices.begin(),usable_devices.end(),"CPU") == usable_devices.end()) {
      // ',CPU' must be deleted
      allDevices.erase(allDevices.find("CPU")-1,4);
    }

    return allDevices;
    }
}

std::vector<std::unique_ptr<ComputeCapability>> OpenVINOExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer,
    const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  bool precision_fp32;
  std::string precision_str = OPENVINO_PRECISION;
  //If precision is specified in python, overwrite build time value (FP32)
  if(requested_precision_.size() != 0) {
    precision_str = requested_precision_;
  }
  if(precision_str == "FP32") {
    precision_fp32 = true;
  }
  else {
    precision_fp32 = false;
  }
  std::string device_id;

  int counter = 0;

  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();

  auto model_proto = GetModelProtoFromFusedNode(graph_viewer);

  std::set<const onnxruntime::NodeArg *> fused_inputs, fused_outputs;
  
  std::string error_msg;
  std::string requested_device;
  
  if(requested_device_.size() > 0) {
    requested_device = requested_device_;
  }
  else {
    requested_device = OPENVINO_DEVICE;
  }

  try {
      device_id = SelectDevice(graph_viewer, requested_device, precision_fp32, error_msg);
  }
  catch (const char* msg) {
    LOGS_DEFAULT(WARNING) << openvino_ep::OpenVINOGraph::log_tag << "Rejecting as graph has unsupported operations." << msg;
    return result;
  }

  precision_str = precision_fp32 ? "FP32" : "FP16";
  LOGS_DEFAULT(INFO) << openvino_ep::OpenVINOGraph::log_tag << "Using device: " << device_id << ", with precision " << precision_str;


  std::string model_proto_strbuf;
  model_proto.SerializeToString(&model_proto_strbuf);

  std::string xml_string, weights_string;

  // Try converting with OpenVINO's Model Optimizer
  try {
    openvino_ep::OpenVINOGraph::ConvertONNXModelToOpenVINOIR(model_proto_strbuf, xml_string, weights_string, precision_fp32);
  } catch (const char* msg) {
    // Model Optimizer cannot convert this model.
    LOGS_DEFAULT(WARNING) << openvino_ep::OpenVINOGraph::log_tag << "Rejecting as Model Optimizer cannot convert this model." << msg;
    return result;
  }

  auto node_indexes = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : node_indexes) {
    sub_graph->nodes.push_back(index);
    const auto node = graph_viewer.GetNode(index);

    // Track graph inputs and initializers
    for (const auto& input_def : node->InputDefs()) {
      if (fused_outputs.find(input_def) == fused_outputs.end()) {
        fused_inputs.insert(input_def);
      } else {
        fused_outputs.erase(input_def);
      }
    }

    // Track graph outputs
    for (const auto& output_def : node->OutputDefs()) {
      if (fused_inputs.find(output_def) == fused_inputs.end()) {
        fused_outputs.insert(output_def);
      } else {
        fused_inputs.erase(output_def);
      }
    }
  }

  ONNX_NAMESPACE::AttributeProto xml_str_attr;
  xml_str_attr.set_name("xml_str");
  xml_str_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
  xml_str_attr.set_s(xml_string);

  ONNX_NAMESPACE::AttributeProto weights_str_attr;
  weights_str_attr.set_name("weights_str");
  weights_str_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
  weights_str_attr.set_s(weights_string);

  ONNX_NAMESPACE::AttributeProto device_id_attr;
  device_id_attr.set_name("device_id");
  device_id_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
  device_id_attr.set_s(device_id);

  ONNX_NAMESPACE::AttributeProto precision_attr;
  precision_attr.set_name("precision");
  precision_attr.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);  
  if(precision_fp32) {
    precision_attr.set_s("FP32");
  }
  else {
    precision_attr.set_s("FP16");
  }

  auto meta_def = onnxruntime::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->attributes["xml_str"] = xml_str_attr;
  meta_def->attributes["weights_str"] = weights_str_attr;
  meta_def->attributes["device_id"] = device_id_attr;
  meta_def->attributes["precision"] = precision_attr;
  meta_def->name = "OpenVINOKernel_" + std::to_string(counter++);
  meta_def->domain = "OpenVINO";
  meta_def->since_version = 1;

  for (auto input : fused_inputs) {
    meta_def->inputs.push_back(input->Name());
  }

  for (auto output : fused_outputs) {
    meta_def->outputs.push_back(output->Name());
  }

  sub_graph->SetMetaDef(meta_def);
  result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));

  LOGS_DEFAULT(INFO) << openvino_ep::OpenVINOGraph::log_tag << "Returning result of GetCapability Function";

  return result;
}

common::Status OpenVINOExecutionProvider::Compile(
    const std::vector<onnxruntime::Node*>& fused_nodes,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (auto fused_node : fused_nodes) {
    std::shared_ptr<openvino_ep::OpenVINOGraph> openvino_graph;
    try {
      openvino_graph = std::make_shared<openvino_ep::OpenVINOGraph>(fused_node);

    } catch (const char* msg) {
      LOGS_DEFAULT(ERROR) << openvino_ep::OpenVINOGraph::log_tag << "Compilation error: " << msg;
      return Status(common::StatusCategory::ONNXRUNTIME,
                    common::StatusCode::NOT_IMPLEMENTED, msg);
    }

    NodeComputeInfo compute_info;

    compute_info.create_state_func =
        [openvino_graph](ComputeContext* context, FunctionState* state) {
          OpenVINOEPFunctionState* p = new OpenVINOEPFunctionState();
          p->allocate_func = context->allocate_func;
          p->destroy_func = context->release_func;
          p->allocator_handle = context->allocator_handle;
          p->openvino_graph = openvino_graph;
          *state = static_cast<FunctionState>(p);
          return 0;
        };

    compute_info.compute_func =
        [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
          Ort::CustomOpApi ort{*api};

          auto function_state = static_cast<OpenVINOEPFunctionState*>(state);

          try {
            function_state->openvino_graph->Infer(ort, context);
          } catch (const char* msg) {
            return common::Status(common::ONNXRUNTIME, common::FAIL);
          }

          return Status::OK();
        };

    compute_info.release_state_func =
        [](FunctionState state) {
          if (state) {
            OpenVINOEPFunctionState* function_state = static_cast<OpenVINOEPFunctionState*>(state);
            delete function_state;
          }
        };

    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}

}  // namespace onnxruntime
