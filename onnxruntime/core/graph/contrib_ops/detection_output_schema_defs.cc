// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "detection_output_schema_defs.h"

#include "core/graph/constants.h"
#include "core/graph/op.h"

namespace onnxruntime {
namespace contrib {

using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::OPTIONAL_VALUE;
using ::ONNX_NAMESPACE::OpSchema;

static const char* DetectionOutput_ver1_doc = R"DOC(
DetectionOutput performs non-maximum suppression to generate the
detection output using information on location and confidence predictions
)DOC";


OpSchema& RegisterDetectionOutputContribOpSchema(OpSchema&& op_schema){
  return op_schema
    .SetDomain(kOnnxDomain)
    .Attr(
        "background_label_id",
        "background label id. If there is no background class, set it to -1",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "code_type",
        "type of coding method for bounding boxes",
        AttributeProto::STRING,
        OPTIONAL_VALUE)
    .Attr(
        "confidence_threshold",
        "only consider detections whose confidences are larger than a threshold."
        "If not provided, consider all boxes",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE)
    .Attr(
        "eta",
        "TBD",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE)
    .Attr(
        "keep_top_k",
        "maximum number of bounding boxes per batch to be kept after NMS step."
        "-1 means keeping all bounding boxes after NMS step",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "nms_threshold",
        "threshold to be used in the NMS stage",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE)
    .Attr(
        "num_classes",
        "number of classes to be predicted",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "share_location",
        "share_location is a flag that denotes if bounding boxes are shared among different classes",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "top_k",
        "maximum number of results to be kept per batch after NMS step."
        "-1 means keeping all bounding boxes",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "variance_encoded_in_target",
        "variance_encoded_in_target is a flag that denotes if variance is encoded in target."
        "If flag is false then it is necessary to adjust the predicted offset accordingly",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .TypeConstraint(
        "T",
        {"tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .Input(
        0,
        "X",
        "2D input tensor with box logits",
        "T")
    .Input(
        1,
        "W",
        "2D input tensor with class predictions",
        "T")
    .Input(
        2,
        "B",
        "3D input tensor with proposals",
        "T")
    .Output(
        0,
        "Y",
        "TBD",
        "T",
        OpSchema::Optional)
    .SetDoc(DetectionOutput_ver1_doc);
}

}  // namespace contrib
}  // namespace onnxruntime
