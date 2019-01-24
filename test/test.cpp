#include<dirent.h>
#include<string.h>
#include<stdio.h>
#include<iostream>
#include<vector>
#include "text_recognization.h"
#include "dlib/dlib/image_io.h"
#include "dlib/dlib/image_processing.h"
#include "dlib/dlib/image_processing/generic_image.h"

int toLowercase(std::string& name_string)
{
   	for (size_t i = 0; i < name_string.size(); i++)
	{
		if (name_string[i] >= 'A' && name_string[i] <= 'Z')
		{
            name_string[i] = name_string[i] + 32;
        }
	}
	return 0;
}
int getAllImages(const char* image_dir, std::vector<std::string>& vImgPaths)
{	
    DIR* pDir = NULL;
	struct dirent* ent = NULL;

	pDir = opendir(image_dir);
	if (pDir == 0)
	{
		printf("open folder(%s) FAIL\n", image_dir);
		return 0;
	}

	std::string strFolder = image_dir;
	if (strFolder[strFolder.size() - 1] != '/')
		strFolder += "/";

	while (ent = readdir(pDir))
	{
		if (ent->d_type & DT_DIR && strcmp(ent -> d_name,".") != 0 && strcmp(ent -> d_name,"..") != 0 )
        {
            char path[255];
            strcpy(path, strFolder.c_str());
            strcat(path, ent->d_name);
            strcat(path, "/");
			getAllImages(path, vImgPaths);
        }
		int len = (int)strlen(ent->d_name);
		if (len >= 5 && strcmp(ent->d_name + len - 4, ".jpg") == 0)
		{
			char filepath[256];
			sprintf(filepath, "%s%s", strFolder.c_str(), ent->d_name);
			vImgPaths.push_back(filepath);
            printf("%s\n", filepath);
		}
	}

	closedir(pDir);

    return 0;
}

int main(int argc, char const *argv[])
{
//    const char* imagePath = argv[1];
//    std::vector<std::string> vec_imagePath;
//    getAllImages(imagePath, vec_imagePath);
    FILE *fp = fopen("annotation_test.txt", "r+");
    if(NULL == fp)
    {
	    fprintf(stderr, "fopen annotation_test.txt error\n");
    }
    char pname[255];
    long idx;
    float count = 0, pre = 0;
    while(1)
    {
        if((fscanf(fp, "%s %ld", pname, &idx)) == EOF)
	    {
//		    fprintf(stderr, "fscanf end(error)\n");
            break;
        }
        dlib::array2d<dlib::rgb_pixel> m;
        load_image(m, pname);
/*        if (m.empty())
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
//        unsigned char* rgbData = new unsigned char[m.cols*m.rows*3];
        fwrite(rgb_panel, 1, m.nc()*m.nr()*3, stream);
		delete [] rgb_panel;
#else
    
        fwrite(&m[0][0], 1, m.nc()*m.nr()*3, stream);
#endif
        fclose(stream);
        DKSBoxTextRecognizationParam  param;
        param.lexicon = false;//使用词典  
        char *result,*finres;

        typedef struct{
	    int xmin; 
	    int ymin; 
	    int xmax; 
	    int ymax;
        }Bbox;
        Bbox bb={0, 0, m.nc(), m.nr()};//124 72 201 89
        DKSBox box = {bb.xmin, bb.ymin, bb.xmax, bb.ymin, bb.xmax, bb.ymax, bb.xmin, bb.ymax};

        DKBoxTextRecognizationInit();

        result = DKBoxTextRecognizationProcess("crnn.data", m.nc(), m.nr(), box, param);
        std::size_t name_sta = std::string(pname).find_first_of("_");
        std::size_t name_end = std::string(pname).find_last_of("_");
        fprintf(stderr, "path:%s\n", pname);
        std::string name_string = std::string(pname).substr(name_sta+1, name_end - name_sta - 1);
        count++;
        printf("%f_recognization results(%s): %s\n",count, name_string.c_str(), result);
        toLowercase(name_string);
        if(strcmp(result, name_string.c_str()) == 0)
            pre++;
        printf("%f_recognization results(%s): %s\n",pre,  name_string.c_str(), result);
        DKBoxTextRecognizationEnd();

    }
    printf("acc:%f\n", pre/count);
    return 0;
}
