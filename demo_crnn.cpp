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
#include <stdio.h>
#include "text_recognization.h"
#include <time.h>
#include "dlib/image_io.h"


int main(int argc, char** argv)
{

    const char* imagepath = argv[1];
    dlib::array2d<dlib::rgb_pixel> m;
    load_image(m, imagepath);
/*    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
*/
    FILE *stream = NULL; 
    stream = fopen("crnn.data", "wb");   

    if(NULL == stream)
    {
        fprintf(stderr, "imgdata read error!");
        exit(1);
    } 

#ifndef RGB_META
    unsigned char* rgb_panel = new unsigned char[m.nc()*m.nr()*3];
    unsigned char* rgb_panel_r = rgb_panel;
    unsigned char* rgb_panel_g = rgb_panel + m.nc()*m.nr();
    unsigned char* rgb_panel_b = rgb_panel + m.nc()*m.nr()*2;    

    for(int r = 0; r < m.nr(); r++)
    {
	for(int c = 0; c < m.nc(); c++)
	{
	    rgb_panel_r[r*m.nc()+c] = m[r][c].red;
	    rgb_panel_g[r*m.nc()+c] = m[r][c].green;
	    rgb_panel_b[r*m.nc()+c] = m[r][c].blue;
	}
    }
//    unsigned char* rgbData = new unsigned char[m.cols*m.rows*3];
    fwrite(rgb_panel, 1, m.nc()*m.nr()*3, stream);
#else
    
    fwrite(&m[0][0], 1, m.nc()*m.nr()*3, stream);
#endif

    fclose(stream);

    clock_t start, finsh;
    const char *rgbfilename = argv[1];
    DKSBoxTextRecognizationParam  param;
    param.lexicon = false;//使用词典  
    char *result,*finres;

    typedef struct{
	int xmin; 
	int ymin; 
	int xmax; 
	int ymax;
    }Bbox;
    Bbox bb={124, 72, 201, 89};//124 72 201 89
    Bbox bb1={135, 96, 189, 111};//124 72 201 89
    DKSBox box = {bb.xmin, bb.ymin, bb.xmax, bb.ymin, bb.xmax, bb.ymax, bb.xmin, bb.ymax};
    DKSBox box1 = {bb1.xmin, bb1.ymin, bb1.xmax, bb1.ymin, bb1.xmax, bb1.ymax, bb1.xmin, bb1.ymax};
//    DKSBox box = {0,0,184,0,184,72,0,72};
//    start = clock();
//    finsh = clock();
//    printf("%ld ms\n", (finsh - start)/1000);
    start = clock();
    DKBoxTextRecognizationInit();
    finsh = clock();
    printf("init cost %d ms\n", (finsh - start)/1000);
    //参数依次为二进制图片文件名、输入图片宽、输入图片高、四边形坐标DKSBox、param.lexicon布尔型变量决定是否用字典。
//    for(int i=0; i<1000; i++)
//    {
//        printf("%dth\n", i);
        start = clock();
        result = DKBoxTextRecognizationProcess("crnn.data", 384, 384, box, param);
        finsh = clock();
        
        printf("recog cost %d ms\n", (finsh - start)/1000);
        printf("recognization results: "); 
        printf("%s\n", result);
//        free(result);
        
//        result = DKBoxTextRecognizationProcess("crnn.data", 384, 384, box1, param);
//        printf("recognization results: "); 
//        printf("%s\n", result);
//        free(result);

//    } 
    DKBoxTextRecognizationEnd();

    printf("recognization results: "); 
    printf("%s\n", result);

    return 0;
}

