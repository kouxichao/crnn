#ifndef TEXT_RECOGNIZATION_H
#define TEXT_RECOGNIZATION_H

typedef struct
{
    //左上角开始顺时针点坐标
    int x1;        
    int y1;
    int x2;
    int y2;
    int x3;
    int y3;
    int x4;
    int y4;   
}DKSBox;

typedef struct
{
	//等待添加
   const char* lexicon;
    
}DKSBoxTextRecognizationParam;


void DKBoxTextRecognizationInit();

char* DKBoxTextRecognizationProcess(const char* imgfilename, DKSBox box, DKSBoxTextRecognizationParam param);

void DKBoxTextRecognizationEnd();

#endif
