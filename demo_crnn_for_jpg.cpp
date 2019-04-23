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

int main(int argc, char** argv)
{
    clock_t start, finsh;
    const char *rgbfilename = argv[1];
    DKSBoxTextRecognizationParam  param;
    param.lexicon = false;//使用词典  
    char *result,*finres;

    DKSBox box = {0,0,1056,0,1056,804,0,804};
//    start = clock();
    DKBoxTextRecognizationInit();
//    finsh = clock();
//    printf("%ld ms\n", (finsh - start)/1000);

    //参数依次为二进制图片文件名、四边形坐标DKSBox，最后一个参数目前没用到。
    result = DKBoxTextRecognizationProcess(rgbfilename, box, param);
    DKBoxTextRecognizationEnd();

    printf("recognization results: "); 
    printf("%s\n", result);

    return 0;
}

