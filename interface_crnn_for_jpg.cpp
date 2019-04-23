#include <stdio.h>
#include "dlib/image_io.h"
#include "net.h"
#include "text_recognization.h"
#include <omp.h>
using namespace std;
static ncnn::Net netCrnn;

void DKBoxTextRecognizationInit()
{
//    clock_t start = clock();
    netCrnn.load_param("crnn.param");
//    clock_t finsh = clock();
//    printf("param : %ld\n", finsh - start);
//    start = clock();
    netCrnn.load_model("crnn.bin");
//    finsh = clock();
//    printf("bin : %ld\n", finsh - start);
     
}

char* DKBoxTextRecognizationProcess(const char* imgfilename, DKSBox box, DKSBoxTextRecognizationParam param)
{
    //裁剪文字区域
    dlib::array2d<dlib::rgb_pixel> rgb_img;
    load_image(rgb_img, imgfilename);
    
    int col = (int)rgb_img.nc();
    int row = (int)rgb_img.nr();

   
    int y_top = box.y1 > box.y2 ? box.y2 : box.y1;
    int y_bottom = box.y3 > box.y4 ? box.y3 : box.y4;
    int x_left = box.x1 > box.x4 ? box.x4 : box.x1;
    int x_right = box.x2 > box.x3 ? box.x2 : box.x3;
    y_top =  y_top > 0 ?  y_top : 0;
    x_left = x_left > 0 ? x_left : 0;
    y_bottom = y_bottom < row ? y_bottom : row; 
    x_right = x_right < col ? x_right : col; 
    col = x_right - x_left;
    row = y_bottom - y_top;
    ncnn::Mat img;
    img.create(col, row, 3, 1);
    
//    clock_t start, finsh;

    #pragma omp parallel for     
    for(int i = y_top; i < y_bottom; i++)
    {
        for(int j=x_left; j<x_right; j++)
        {
            *((unsigned char*)(img.data)+3*(i-y_top)*col+3*(j-x_left))   = rgb_img[i][j].blue;
            *((unsigned char*)(img.data)+3*(i-y_top)*col+3*(j-x_left)+1) = rgb_img[i][j].green;
            *((unsigned char*)(img.data)+3*(i-y_top)*col+3*(j-x_left)+2) = rgb_img[i][j].red;
        }
    }

    //预处理并获取字符序列索引
    ncnn::Mat in,input_data;
    ncnn::Mat pred;
    in = ncnn::Mat::from_pixels((unsigned char*)img.data, ncnn::Mat::PIXEL_BGR2GRAY, col, row);
    ncnn::resize_bilinear(in,input_data,100,32);
    input_data.reshape(100,32,1);
    
    #pragma omp parallel for
    for(int i=0; i<100 * 32; i++)
    {
        *((float*)input_data.data+i) = ((*((float*)input_data.data+i))/255.f - 0.5)/0.5;
    }

//    start = clock();    
    ncnn::Extractor ex = netCrnn.create_extractor();
    ex.set_num_threads(4);
    ex.set_light_mode(false);
    ex.input("data", input_data);
    ex.extract("preds", pred);

//    finsh = clock();
//    cout << "extract fe cost " << (finsh - start)/1000 << "ms" << endl;             
    //对输出字符索引解码得到字符串。
    float maxprob; 
    int pre_index = 0;
    char alphabet[] = "0123456789abcdefghijklmnopqrstuvwxyz"; 
    static std::vector<char> result;
    for (int j=0; j<pred.h; j++)
    {
        int char_index = 0;
        maxprob = *((float*)pred.row(j));

        for (int i=0; i<pred.w; i++)
        {
            if (*((float*)pred.row(j)+i) > maxprob)
            {
                maxprob = *((float*)pred.row(j)+i);
                char_index = i;
            }
        }
	printf("%f\n", maxprob);
	printf("%c\n", alphabet[char_index-1]);
        if(char_index != 0 && (j==0 || char_index != pre_index))
        { 
//	    printf("%c\n", alphabet[char_index-1]);
            result.push_back(alphabet[char_index-1]);            
        }      
        pre_index = char_index;
    }
    result.push_back('\0');

    if(param.lexicon)
    {
        return minDistanceWord((char*)result.data());
    }
    else
    {
        return result.data();
    }
}

void DKBoxTextRecognizationEnd()
{
    netCrnn.clear();	
}
