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

    int xc_size = weight_xc_data_size / num_output / 4;
    int hc_size = weight_hc_data_size / num_output / 4;

    weight_xc_data.resize(num_lstm_layer*numdir);
    bias_xc_data.resize(num_lstm_layer*numdir);
    weight_hc_data.resize(num_lstm_layer*numdir);
    bias_hc_data.resize(num_lstm_layer*numdir);

    for(int i=0; i<numdir; i++)
    {
    // raw weight data
        for(int l=0; l<num_lstm_layer; l++)
        {
            if(l == 0)
            {   //    fprintf(stderr,"________%d", l);
                weight_xc_data[i*num_lstm_layer+l] = mb.load(xc_size * 4, num_output, 1);
                if (weight_xc_data.empty())
                return -100;
            }
            else
            {
                weight_xc_data[i*num_lstm_layer+l] = mb.load(hc_size * 4, num_output, 1);
                if (weight_xc_data.empty())
                return -100;
            }
    	
            bias_xc_data[i*num_lstm_layer+l] = mb.load(4, num_output, 1);
    	    if (bias_xc_data.empty())
                return -100;

    	    weight_hc_data[i*num_lstm_layer+l] = mb.load(hc_size * 4, num_output, 1);
    	    if (weight_hc_data.empty())
                return -100;
   
            bias_hc_data[i*num_lstm_layer+l] = mb.load(4, num_output, 1);
            if (bias_hc_data.empty())
                return -100;
       }
      // fprintf(stderr,"________%d", hc_size);
        
    }


    return 0;
}

int BiLSTM::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{

    // size x 1 x T
    const Mat& input_blob = bottom_blobs[0];

    // T, 0 or 1 each
    const Mat& cont_blob = bottom_blobs[1];

    int T = input_blob.c;
    //int size = input_blob.w;
    int numdir = isbilstm ? 2 : 1;

    // initial hidden state
    Mat hidden(num_output, num_lstm_layer);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    // internal cell state
    Mat cell(num_output, num_lstm_layer);
    if (cell.empty())
        return -100;
    // 4 x num_output
    Mat gates(4, num_output);
    if (gates.empty())
        return -100;

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output*numdir, 1, T);
    if (top_blob.empty())
        return -100;

    // unroll
    for(int d=0; d<numdir; d++)
    {    
    for (int t=0; t<T; t++)
    {                     fprintf(stderr, "---%d", d);                                                                           
        // clip hidden by continuation indicator
        // h_cont_{t-1} = cont_t * h_{t-1}
        // h_cont_{t-1} = h_{t-1} if cont_t == 1
        //                0       otherwise
        // calculate hidden
        // gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
        float* hidden_data = hidden;  
        Mat x;
        int size;
        for (int l=0; l<num_lstm_layer; l++)
        {
            const float cont = cont_blob[T*l+t];
            if (l == 0 && d == 0)
            {
                x = input_blob.channel(t);
                size = input_blob.w;

            }
            else if(l == 0 && d != 0)
            {
                x = input_blob.channel(T-t-1);
                size = input_blob.w;             
            }
            else 
            {
                x.data = hidden_data+num_output*(l-1);
                size = num_output;
                                                          // fprintf(stderr,"\n%d",size);
            }
           
           int num_layer = numdir ? (l+2) : l;
           for (int q=0; q<num_output; q++)
           {
               float h_cont = cont ? hidden_data[num_output*l+q] : 0.f;

               const float* x_data = x;
           
               const float* bias_xc_data_ptr = (const float*)bias_xc_data[num_layer] + 4*q;
               const float* bias_hc_data_ptr = (const float*)bias_hc_data[num_layer] + 4*q;
            
               float* gates_data = (float*)gates + 4 * q;

            // gate I F O G
               const float* weight_hc_data_I = (const float*)weight_hc_data[num_layer] + weight_hc_data[num_layer].w * q;
               const float* weight_xc_data_I = (const float*)weight_xc_data[num_layer] + weight_xc_data[num_layer].w * q;
               const float* weight_hc_data_F = (const float*)weight_hc_data[num_layer] + weight_hc_data[num_layer].w * q + num_output;
               const float* weight_xc_data_F = (const float*)weight_xc_data[num_layer] + weight_xc_data[num_layer].w * q + size;
               const float* weight_hc_data_O = (const float*)weight_hc_data[num_layer] + weight_hc_data[num_layer].w * q + num_output*2;
               const float* weight_xc_data_O = (const float*)weight_xc_data[num_layer] + weight_xc_data[num_layer].w * q + size*2;
               const float* weight_hc_data_G = (const float*)weight_hc_data[num_layer] + weight_hc_data[num_layer].w * q + num_output*3;
               const float* weight_xc_data_G = (const float*)weight_xc_data[num_layer] + weight_xc_data[num_layer].w * q + size*3;

               float I = bias_hc_data_ptr[0]+bias_xc_data_ptr[0];
               float F = bias_hc_data_ptr[1]+bias_xc_data_ptr[1];
               float O = bias_hc_data_ptr[2]+bias_xc_data_ptr[2];
               float G = bias_hc_data_ptr[3]+bias_xc_data_ptr[3];

               for (int i=0; i<num_output; i++)
               {
                  I += weight_hc_data_I[i] * h_cont; 
                  F += weight_hc_data_F[i] * h_cont; 
                  O += weight_hc_data_O[i] * h_cont; 
                  G += weight_hc_data_G[i] * h_cont; 
               }

              for(int i=0; i<size; i++)
              {
                                                       //      fprintf(stderr,"%d.....",size);
                  G += weight_xc_data_G[i] * x_data[i];
                  O += weight_xc_data_O[i] * x_data[i];
                  F += weight_xc_data_F[i] * x_data[i];
                  I += weight_xc_data_I[i] * x_data[i];
              } 
                                                      //    fprintf(stderr,"---%d",q);
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
            float* cell_data = cell;
            Mat output = top_blob.channel(t);
            float* output_data = output;
            for (int q=0; q<num_output; q++)
            {
                float* gates_data = (float*)gates + 4 * q;

                float I = gates_data[0];
                float F = gates_data[1];
                float O = gates_data[2];
                float G = gates_data[3];

                I = 1.f / (1.f + exp(-I));
                F = cont ? 0.f : 1.f / (1.f + exp(-F));
                O = 1.f / (1.f + exp(-O));
                G = tanh(G);

                float cell = F * cell_data[q] + I * G;
                float H = O * tanh(cell);

                cell_data[num_output*l+q] = cell;
                hidden_data[num_output*l+q] = H;
  
                if(d == 0 && l == num_lstm_layer-1)
                output_data[q] = H; 
                else if(d == 1 && l == num_lstm_layer-1)
                output_data[num_output*numdir-q-1] = H; 
            }
        }  

        // no cell output here
    }
    }
    return 0;
}

} // namespace ncnn
