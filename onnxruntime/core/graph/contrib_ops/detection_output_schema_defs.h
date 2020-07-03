// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {
namespace contrib {

::ONNX_NAMESPACE::OpSchema& RegisterDetectionOutputContribOpSchema(::ONNX_NAMESPACE::OpSchema&& op_schema);

}
}
