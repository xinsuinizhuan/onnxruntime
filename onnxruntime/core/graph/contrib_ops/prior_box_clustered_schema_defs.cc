// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "prior_box_clustered_schema_defs.h"

#include "core/graph/constants.h"
#include "core/graph/op.h"

namespace onnxruntime {
namespace contrib {

using ::ONNX_NAMESPACE::AttributeProto;
using ::ONNX_NAMESPACE::OPTIONAL_VALUE;
using ::ONNX_NAMESPACE::OpSchema;

static const char* PriorBoxClustered_ver1_doc = R"DOC(

PriorBoxClustered operation generates prior boxes of
specified sizes normalized to the input image size.

)DOC";


OpSchema& RegisterPriorBoxClusteredContribOpSchema(OpSchema&& op_schema){
  return op_schema
    .SetDomain(kOnnxDomain)
    .Attr(
        "width",
        "width specifies desired boxes widths in pixels.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE)
    .Attr(
        "height",
        "height specifies desired boxes heights in pixels.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE)
    .Attr(
        "clip",
        "clip is a flag that denotes if each value in the"
        "output tensor should be clipped within [0,1]",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "flip",
        "flip is a flag that denotes that each aspect_ratio is duplicated and flipped.",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "step",
        "step is a distance between box centers. For example, step equal 85 means that"
        "the distance between neighborhood prior boxes centers is 85. If both step_h"
        "and step_w are 0 then they are updated with value of step. If after that they"
        "are still 0 then they are calculated as input image width(height) divided with"
        "first input width(height).",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE)
    .Attr(
        "step_w",
        "TBD",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE)
    .Attr(
        "step_h",
        "TBD",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE)
    .Attr(
        "img_w",
        "img_w specifies width of input image. These attributes are taken from the second"
        "input image_size width unless provided explicitly as the value for this attributes.",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "img_h",
        "img_h specifies height of input image. These attributes are taken from the second"
        "input image_size height unless provided explicitly as the value for this attributes.",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "img_size",
        "TBD",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "offset",
        "offset is a shift of box respectively to top left corner. For example, "
        "offset equal 85 means that the shift of neighborhood prior boxes centers is 85.",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE)
    .Attr(
        "variance",
        "variance denotes a variance of adjusting bounding boxes.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE)
    .TypeConstraint(
        "T",
        {"tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .Input(
        0,
        "X",
        "1D tensor with two integer elements [height, width]. Specifies the spatial"
        "size of generated grid with boxes",
        "T")
    .Input(
        1,
        "W",
        "1D tensor with two integer elements [image_height, image_width]"
        "that specifies shape of the image for which boxes are generated",
        "T")
    .Output(
        0,
        "Y",
        "2D tensor of shape [2, 4 * height * width * priors_per_point] with box coordinates."
        "The priors_per_point is the number of boxes generated per each grid element."
        "The number depends on layer attribute values",
        "T",
        OpSchema::Optional)
    .SetDoc(PriorBoxClustered_ver1_doc);
}

}  // namespace contrib
}  // namespace onnxruntime
