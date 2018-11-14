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

#include "bilstm.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(BiLSTM)

BiLSTM::BiLSTM()
{
    one_blob_only = false;
    support_inplace = false;
}

int BiLSTM::load_param(const ParamDict& pd)
{
    num_lstm_layer = pd.get(0, 0);
    isbilstm = pd.get(1, 0);
    num_output = pd.get(2, 0);
    weight_xc_data_size = pd.get(3, 0);
    weight_hc_data_size = pd.get(4, 0);
    return 0;
}

int BiLSTM::load_model(const ModelBin& mb)
{
    int numdir = isbilstm ? 2 : 1;

    weight_xc_data.resize(num_lstm_layer*numdir);
    bias_xc_data.resize(num_lstm_layer*numdir);
    weight_hc_data.resize(num_lstm_layer*numdir);
    bias_hc_data.resize(num_lstm_layer*numdir);

    for(int i=0; i<numdir; i++)
    {
        for(int l=0; l<num_lstm_layer; l++)
        {
            if(l == 0)
            {   
                weight_xc_data[i*num_lstm_layer+l] = mb.load(weight_xc_data_size, 0);
                if (weight_xc_data.empty())
                return -100;
            }
            else
            {
                weight_xc_data[i*num_lstm_layer+l] = mb.load(weight_hc_data_size, 0);
                if (weight_xc_data.empty())
                return -100;
            }
    	
            bias_xc_data[i*num_lstm_layer+l] = mb.load(num_output*4, 1);
    	    if (bias_xc_data.empty())
                return -100;

    	    weight_hc_data[i*num_lstm_layer+l] = mb.load(weight_hc_data_size, 0);
    	    if (weight_hc_data.empty())
                return -100;
   
            bias_hc_data[i*num_lstm_layer+l] = mb.load(num_output*4, 1);
            if (bias_hc_data.empty())
                return -100;
       }     
    }

    return 0;
}

int BiLSTM::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    const Mat& input_blob = bottom_blobs[0];
//    const Mat& cont_blob = bottom_blobs[1];
    
    int T = input_blob.c;
    int numdir = isbilstm ? 2 : 1;

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output*numdir, T);
    if (top_blob.empty())
        return -100;

    for(int d=0; d<numdir; d++)
    { 
        Mat hidden(num_output, num_lstm_layer);
        if (hidden.empty())
            return -100;
        hidden.fill(0.f);

        Mat cell(num_output, num_lstm_layer);
        if (cell.empty())
            return -100;
        cell.fill(0.f);

        Mat gates(4, num_output);
        if (gates.empty())
            return -100;      
                  
        float* hidden_data = hidden;
        float* cell_data = cell;

        Mat x;
        int size;

        for (int t=0; t<T; t++)
        {                
            for (int l=0; l<num_lstm_layer; l++)
            {
       //         const  float cont = cont_blob[T*l+t];
                if (l == 0 && d == 0)
                {
                    x = input_blob.channel(t);
                    size = input_blob.w;
                }
                else if(l == 0 && d == 1)
                {
                    x = input_blob.channel(T-t-1);
                    size = input_blob.w;             
                }
                else 
                { 
                    x = hidden;
                    size = num_output;
                } 
               
                int num_layer;
                if(num_lstm_layer == 2)
                    num_layer = d ? (l+2) : l;
                else 
                    num_layer = d ? 1 : 0;

                for (int q=0; q<num_output; q++)
                {
                //   float h_cont = cont ? hidden_data[num_output*l+q] : 0.f;
                    const float* x_data = x;
                    const float* bias_xc_data_ptr = (const float*)bias_xc_data[num_layer];
                    const float* bias_hc_data_ptr = (const float*)bias_hc_data[num_layer];
                    float* gates_data = (float*)gates + 4 * q;
               
                    const float* weight_hc_data_I = (const float*)weight_hc_data[num_layer] + num_output * q;
                    const float* weight_xc_data_I = (const float*)weight_xc_data[num_layer] + size * q;
                    const float* weight_hc_data_F = (const float*)weight_hc_data[num_layer] + num_output * q + weight_hc_data[num_layer].w/4;
                    const float* weight_xc_data_F = (const float*)weight_xc_data[num_layer] + size * q + weight_xc_data[num_layer].w /4 ;
                    const float* weight_hc_data_O = (const float*)weight_hc_data[num_layer] + num_output * q + weight_hc_data[num_layer].w * 3 /4;
                    const float* weight_xc_data_O = (const float*)weight_xc_data[num_layer] + size * q + weight_xc_data[num_layer].w * 3 / 4 ;
                    const float* weight_hc_data_G = (const float*)weight_hc_data[num_layer] + num_output * q + weight_hc_data[num_layer].w/2;
                    const float* weight_xc_data_G = (const float*)weight_xc_data[num_layer] + size * q + weight_xc_data[num_layer].w/2;

                    float I = bias_hc_data_ptr[q] + bias_xc_data_ptr[q];
                    float F = bias_hc_data_ptr[q+num_output*2] + bias_xc_data_ptr[q+num_output];
                    float O = bias_hc_data_ptr[q+num_output*3] + bias_xc_data_ptr[q+num_output*3];
                    float G = bias_hc_data_ptr[q+num_output] + bias_xc_data_ptr[q+num_output*2];

                    for (int i=0; i<num_output; i++)
                    {
                  //      float h_cont = cont ? hidden_data[num_output*l+i] : 0.f;
                        float h_cont = hidden_data[num_output*l+i];

                        I += weight_hc_data_I[i] * h_cont; 
                        F += weight_hc_data_F[i] * h_cont; 
                        O += weight_hc_data_O[i] * h_cont; 
                        G += weight_hc_data_G[i] * h_cont; 
                    }

                    for(int i=0; i<size; i++)
                    {
                        G += weight_xc_data_G[i] * x_data[i];
                        O += weight_xc_data_O[i] * x_data[i];
                        F += weight_xc_data_F[i] * x_data[i];
                        I += weight_xc_data_I[i] * x_data[i];
                    } 

                    gates_data[0] = I;
                    gates_data[1] = F;
                    gates_data[2] = O;
                    gates_data[3] = G;
              
                }

                // lstm unit
                // sigmoid(I)
                // sigmoid(F)
                // sigmoid(O)
                // tanh(G)
                // c_t := f_t .* c_{t-1} + i_t .* g_t
                // h_t := o_t .* tanh[c_t]
                int tt;
                if(d == 1)
                    tt = T - t -1; 
                else
                    tt = t;
    
                float* output_data = top_blob.row(tt);
                
                for (int q=0; q<num_output; q++)
                {
                    float* gates_data = (float*)gates + 4 * q;

                    float I = gates_data[0];
                    float F = gates_data[1];
                    float O = gates_data[2];
                    float G = gates_data[3];
             
                    I = 1.f / (1.f + exp(-I));
                 //   F = cont ? 1.f / (1.f + exp(-F)) : 0.f;
                    F = 1.f / (1.f + exp(-F));
                    O = 1.f / (1.f + exp(-O));
                    G = tanh(G);
               
                    float cell = F * cell_data[num_output*l+q] + I * G;
                    float H = O * tanh(cell);

                    cell_data[num_output*l+q] = cell;
                    hidden_data[num_output*l+q] = H;
                 
                    if(l == num_lstm_layer-1)
                        output_data[num_output*d+q] = H;
                }
            } 
        }
    }

    return 0;
}

} // namespace ncnn
