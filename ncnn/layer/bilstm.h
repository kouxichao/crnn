// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_BILSTM_H
#define LAYER_BILSTM_H

#include "layer.h"

namespace ncnn {

class BiLSTM : public Layer
{
public:
    BiLSTM();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;

public:
    // param
    int num_lstm_layer;
    int isbilstm;
    int num_output;
    int weight_xc_data_size;
    int weight_hc_data_size;

    // model
    std::vector<Mat> weight_xc_data;
    std::vector<Mat> weight_hc_data;
    std::vector<Mat> bias_xc_data;
    std::vector<Mat> bias_hc_data;
};

} // namespace ncnn

#endif // LAYER_LSTM_H
