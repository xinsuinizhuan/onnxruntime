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

// This Doc based on LSTM_ver7, and modification
static const char* PriorBoxClustered_ver1_doc = R"DOC(
Computes an one-layer RNN where its RNN Cell is an AttentionWrapper wrapped a LSTM Cell. The RNN layer
contains following basic component: LSTM Cell, Bahdanau Attention Mechanism, AttentionWrapp.

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

  Softmax(x)             - exp(x) / sum(exp(x))

Bahdanau Attention Mechanism:
    `M` -  Memory tensor.

    `VALUES` - masked Memory by its real sequence length.

    `MW` - Memory layer weight.

    `KEYS` - Processed memory tensor by the memory layer.
             KEYS = M * MW

    `Query` - Query tensor, normally at specific time step in sequence.

    `QW` - Query layer weight in the attention mechanism

    `PQ` - processed query,  = `Query` * `QW`

    `V' - attention vector

    `ALIGN` - calculated alignment based on Query and KEYS
        ALIGN = softmax(reduce_sum(`V` * Tanh(`KEYS` + `PQ`)))

    `CONTEXT` - context based on `ALIGN` and `VALUES`
        CONTEXT = `ALIGN` * `VALUES`


LSTM Cell:
  `X` - input tensor concat with attention state in the attention wrapper

  `i` - input gate

  `o` - output gate

  `f` - forget gate

  `c` - cell gate

  `t` - time step (t-1 means previous time step)

  `W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates

  `R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates

  `Wb[iofc]` - W bias vectors for input, output, forget, and cell gates

  `Rb[iofc]` - R bias vectors for input, output, forget, and cell gates

  `P[iof]`  - P peephole weight vector for input, output, and forget gates

  `WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates

  `RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates

  `WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates

  `RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates

  `PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

  `H` - Hidden state

  `num_directions` - 2 if direction == bidirectional else 1

  Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

    - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)

    - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)

    - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

    - Ct = ft (.) Ct-1 + it (.) ct

    - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)

    - Ht = ot (.) h(Ct)


AttentionWrapp Notations:
  `lstm()' - wrapped inner cell.
           Ht, Ct = lstm(concat(Xt, ATTNt-1), Ct-1)

  `am()` - attention mechanism the wrapper used.
           CONTEXTt, ALIGNt = am(Ht, ALIGNt-1)

  `AW` - attention layer weights, optional.

  `ATTN` - attention state, initial is zero. If `AW` provided, it is the output of the attention layer,
                ATTNt = concat(Ht, CONTEXTt) * AW
           otherwise,
                ATTNt = CONTEXTt

RNN layer output:
  `Y` - if needed is the sequence of Ht from lstm cell.

  `Y_h` - is the last valid H from lstm cell.

  `Y_c` - is the last valid C from lstm cell.

)DOC";


OpSchema& RegisterPriorBoxClusteredContribOpSchema(OpSchema&& op_schema){
  return op_schema
    .SetDomain(kOnnxDomain)
    .Attr(
        "width",
        "A list of 3 (or 6 if bidirectional) activation functions "
        "for input, output, forget, cell, and hidden. The activation functions must "
        "be one of the activation functions specified above. Optional: See the equations "
        "for default if not specified.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE)
    .Attr(
        "height",
        "A list of 3 (or 6 if bidirectional) activation functions "
        "for input, output, forget, cell, and hidden. The activation functions must "
        "be one of the activation functions specified above. Optional: See the equations "
        "for default if not specified.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE)
    .Attr(
        "clip",
        "Optional scaling values used by some activation functions. The values are consumed "
        "in the order of activation functions, for example (f, g, h) in LSTM. Default values "
        "are the same as of corresponding ONNX operators.For example with LeakyRelu, the "
        "default alpha is 0.01.",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "flip",
        "Optional scaling values used by some activation functions. The values are consumed "
        "in the order of activation functions, for example (f, g, h) in LSTM. Default values "
        "are the same as of corresponding ONNX operators.For example with LeakyRelu, the "
        "default alpha is 0.01.",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "step",
        "Optional scaling values used by some activation functions. The values are consumed in "
        "the order of activation functions, for example (f, g, h) in LSTM. Default values are "
        "the same as of corresponding ONNX operators.",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE)
    .Attr(
        "step_w",
        "Optional scaling values used by some activation functions. The values are consumed in "
        "the order of activation functions, for example (f, g, h) in LSTM. Default values are "
        "the same as of corresponding ONNX operators.",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE)
    .Attr(
        "step_h",
        "Optional scaling values used by some activation functions. The values are consumed in "
        "the order of activation functions, for example (f, g, h) in LSTM. Default values are "
        "the same as of corresponding ONNX operators.",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE)
    .Attr(
        "img_w",
        "Optional scaling values used by some activation functions. The values are consumed in "
        "the order of activation functions, for example (f, g, h) in LSTM. Default values are "
        "the same as of corresponding ONNX operators.",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "img_h",
        "Optional scaling values used by some activation functions. The values are consumed in "
        "the order of activation functions, for example (f, g, h) in LSTM. Default values are "
        "the same as of corresponding ONNX operators.",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "img_size",
        "Optional scaling values used by some activation functions. The values are consumed in "
        "the order of activation functions, for example (f, g, h) in LSTM. Default values are "
        "the same as of corresponding ONNX operators.",
        AttributeProto::INT,
        OPTIONAL_VALUE)
    .Attr(
        "offset",
        "Cell clip threshold. Clipping bounds the elements of a tensor in the range of "
        "[-threshold, +threshold] and is applied to the input of activations. No clip if not "
        "specified.",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE)
    .Attr(
        "variance",
        "Couple the input and forget gates if 1, default 0.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE)
    .TypeConstraint(
        "T",
        {"tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.")
    .Input(
        0,
        "X",
        "The input sequences packed (and potentially padded) into one 3-D tensor "
        "with the shape of `[seq_length, batch_size, input_size]`",
        "T")
    .Input(
        1,
        "W",
        "The weight tensor for the gates. Concatenation of `W[iofc]` and "
        "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
        "`[num_directions, 4*hidden_size, input_size]`.",
        "T")
    .Output(
        0,
        "Y",
        "A tensor that concats all the intermediate output values of the hidden. "
        "It has shape `[seq_length, num_directions, batch_size, hidden_size]`",
        "T",
        OpSchema::Optional)
    .SetDoc(PriorBoxClustered_ver1_doc);
}

}  // namespace contrib
}  // namespace onnxruntime
