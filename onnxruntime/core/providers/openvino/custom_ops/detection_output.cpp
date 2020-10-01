//*****************************************************************************
// Copyright 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************


#include "detection_output.hpp"
#include "ngraph/op/detection_output.hpp"
#include "ngraph/node.hpp"
#include <ngraph/frontend/onnx_import/default_opset.hpp>
#include <ngraph/frontend/onnx_import/core/node.hpp>

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector detection_output(const Node& node)
                {

                  auto inputs = node.get_ng_inputs();
                  auto box_logits = inputs[0];
                  auto class_preds = inputs[1];
                  auto proposals = inputs[2];

                  ngraph::op::DetectionOutputAttrs attrs;
                  attrs.num_classes = node.get_attribute_value<int64_t>("num_classes");
                  attrs.background_label_id = node.get_attribute_value<int64_t>("background_label_id", 0);
                  attrs.top_k = node.get_attribute_value<int64_t>("top_k", -1);
                  attrs.variance_encoded_in_target = node.get_attribute_value<int64_t>("variance_encoded_in_target", 0);
                  attrs.keep_top_k = {static_cast<int>(node.get_attribute_value<int64_t>("keep_top_k", 1))};
                  auto code_type = node.get_attribute_value<std::string>(
                                    "code_type", std::string{"caffe.PriorBoxParameter.CORNER"});
                  if (code_type.find("caffe.PriorBoxParameter.") == std::string::npos) {
                      code_type = "caffe.PriorBoxParameter." + code_type;
                  }
                  attrs.code_type = code_type;
                  attrs.share_location = node.get_attribute_value<int64_t>("share_location", 1);
                  attrs.nms_threshold = node.get_attribute_value<float>("nms_threshold");
                  attrs.confidence_threshold = node.get_attribute_value<float>("confidence_threshold", 0);
                  return {std::make_shared<default_opset::DetectionOutput>(
                           box_logits, class_preds, proposals, attrs)};
                }
            } // namespace set_1

        } //namespace op

    } // namespace onnx_import
} // namespace ngraph