// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "prior_box_schema_defs.h"

#include "core/graph/constants.h"
#include "core/graph/op.h"

namespace onnxruntime {
namespace contrib {

using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::OPTIONAL_VALUE;
using ::ONNX_NAMESPACE::OpSchema;

static const char* PriorBox_ver1_doc = R"DOC(

PriorBox operation generates prior boxes of specified sizes
and aspect ratios across all dimensions.

)DOC";


OpSchema& RegisterPriorBoxContribOpSchema(OpSchema&& op_schema){
  return op_schema
    .SetDomain(kOnnxDomain)
    .Attr(
        "min_size",
        "min_size is the minimum box size (in pixels)."
        "For example, min_size equal 15 means that the minimum box size is 15.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE)
    .Attr(
        "max_size",
        "max_size is the maximum box size (in pixels)."
        "For example, max_size equal 15 means that the maximum box size is 15."
        "A list of 3 (or 6 if bidirectional) activation functions ",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE)
    .Attr(
        "aspect_ratio",
        "aspect_ratio is a variance of aspect ratios."
        "Duplicate values are ignored. For example, aspect_ratio equal '2.0,3.0' means that"
        "for the first box aspect_ratio is equal to 2.0 and for the second box is 3.0.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE)
    .Attr(
        "flip",
        "flip is a flag that denotes that each aspect_ratio is duplicated and flipped."
        "For example, flip equals 1 and aspect_ratio equals to '4.0,2.0' mean that "
        "aspect_ratio is equal to '4.0,2.0,0.25,0.5' ",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "clip",
        "clip is a flag that denotes if each value in"
        "the output tensor should be clipped to [0,1] interval.",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "step",
        "step is a distance between box centers. For example, step equal 85 means "
        "that the distance between neighborhood prior boxes centers is 85.",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE)
    .Attr(
        "offset",
        "offset is a shift of box respectively to top left corner. For example,"
        "offset equal 85 means that the shift of neighborhood prior boxes centers is 85.",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE)
    .Attr(
        "variance",
        "variance denotes a variance of adjusting bounding boxes."
        "The attribute could contain 0, 1 or 4 elements.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE)
    .Attr(
        "scale_all_sizes",
        "scale_all_sizes is a flag that denotes type of inference. For example, "
        "scale_all_sizes equals 0 means that the PriorBox layer is inferred in "
        "MXNet-like manner. In particular, max_size attribute is ignored",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "fixed_ratio",
        "fixed_ratio is an aspect ratio of a box. For example, "
        "fixed_ratio equal to 2.000000 means that the aspect ratio for "
        "the first box aspect ratio is 2.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE)
    .Attr(
        "fixed_size",
        "fixed_size is an initial box size (in pixels). For example, "
        "fixed_size equal to 15 means that the initial box size is 15.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE)
    .Attr(
        "density",
        " density is the square root of the number of boxes of each type. For example,"
        "density equal to 2 means that the first box generates four boxes of the same size",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE)
    .TypeConstraint(
        "T",
        {"tensor(float)"},
        "Constrain input and output types to float tensors.")
    .Input(
        0,
        "output_size",
        "1D tensor with two integer elements [height, width]."
        "Specifies the spatial size of generated grid with boxes.",
        "T")
    .Input(
        1,
        "image_size",
        "1D tensor with two integer elements [image_height, image_width] "
        "that specifies shape of the image for which boxes are generated.",
        "T")
    .Output(
        0,
        "Y",
        "2D tensor of shape [2, 4 * height * width * priors_per_point] with box coordinates."
        "The priors_per_point is the number of boxes generated per each grid element. The number depends on layer attribute values.",
        "T",
        OpSchema::Optional)
    .SetDoc(PriorBox_ver1_doc);
}

}  // namespace contrib
}  // namespace onnxruntime
