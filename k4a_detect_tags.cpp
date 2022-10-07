#include <cstdio>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <string>
#include <algorithm>
#include <cstdint>
#include <vector>
#include <Eigen/Eigen>
// OpenCV header files.
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// Apriltag header files
#include <apriltag.h>
#include <tagStandard41h12.h>
//#include <tag36h11.h>
#include <apriltag_pose.h>

class TagDetector
{
    public:
        apriltag_family_t *tf;
        apriltag_detector_t* td;

        TagDetector()
        {
            tf = tagStandard41h12_create();
            //tf = tag36h11_create();
            td = apriltag_detector_create();
            apriltag_detector_add_family_bits(td, tf, 1);
            td->quad_decimate = 1.f;
            td->quad_sigma = 0.8f;
            td->nthreads = 8;
            td->refine_edges = 1;
            td->decode_sharpening = 0.25;
        }

        ~TagDetector()
        {
            tagStandard41h12_destroy(tf);
            //tag36h11_destroy(tf); 
            apriltag_detector_destroy(td);
        }
}; // End class def - TagDetector

void validate_input_files(char* argv[], int argc, std::vector<std::string>& filenames)
{
    
    // this is to ensure that all the supplied files terminate with a '.mkv' extension
    for(int i{1}; i < argc; i++){
        std::string tmp = argv[i];
        size_t pos = tmp.rfind('.');

        if (pos == std::string::npos){
            std::cout << "'" << tmp << "' is not a valid file, ignoring..." << std::endl;
            continue;
        }   
        
        std::string ext = tmp.substr(pos+1);

        if (ext == "mkv"){
            filenames.emplace_back(argv[i]);
        }
    }

}

void create_output_fnames(const std::vector<std::string>& filenames, std::vector<std::string>& output_filenames ){

    // replace .mkv with .ply
        for( const auto& filename : filenames){
            size_t lastindex = filename.find_last_of("."); 
            std::string rawname = filename.substr(0, lastindex); 
            output_filenames.emplace_back(rawname + ".ply");
        }
}


int get_camID(const std::string& filename){
    // gets the camID as the digit right before the '.' character
    size_t lastindex = filename.find_last_of(".");
    std::string rawname = filename.substr(0, lastindex);
    int camID = static_cast<int>(rawname[rawname.size()-1]);
    return camID; 
}

template<typename T>
void print_vector(const std::vector<T>& vector,const std::string& delim)
{
    for(const auto& item : vector){
        std::cout << item << delim;
    }
    std::cout << std::endl;
}



// ********************** BEGIN MAIN FUNCTION **********************//
int main(int argc, char* argv[])
{
    int returnCode = 1;
    if(argc < 2){
        std::cout << "\tUsage : " << argv[0] << " <view1.mkv> <view2.mkv> ..." << std::endl;
        return -1;
    }
    std::vector<std::string> filenames, output_filenames;
    std::vector<char>stars(100,'=');

    validate_input_files(argv, argc,filenames);

    if(filenames.empty()){
        std::cout << "Failed to provide valid input files. Exiting program..." << std::endl;
        return -1;
    }

    print_vector(stars, "");
    std::cout << "Detecting Apriltags for : \n\t";
    print_vector(filenames," ");
    print_vector(stars,"");
    create_output_fnames(filenames, output_filenames);
    //return 0;

    for(int i{0}; i < filenames.size(); i++){ // For each file, open the recording and detect the tag in the color image frame.
        
        std::string inputFname = filenames[i];
        int camID = get_camID(inputFname);
       
        k4a_device_t device = NULL;
        const int32_t TIMEOUT_IN_MS = 1000;
        k4a_image_t color_image = NULL;
        k4a_capture_t capture = NULL;
        k4a_calibration_t calibration;

        /* Read data from Capture MKV file */
        k4a_playback_t playback = NULL;
        k4a_result_t result;
        k4a_stream_result_t stream_result = K4A_STREAM_RESULT_SUCCEEDED; // init to success state and watch for change later

        // this opens the recording stored in inputFname
        result = k4a_playback_open(inputFname.c_str(),&playback);
        
        if(result != K4A_RESULT_SUCCEEDED || playback == NULL){
            std::cout<<"Failed to open the recording: " << inputFname << std::endl;
            
            if(playback != NULL){
                k4a_playback_close(playback);
            }
            return -1;
        }
        if (K4A_RESULT_SUCCEEDED != k4a_playback_get_calibration(playback, &calibration)){
            std::cout << "Failed to get calibration" << std::endl;
            if(playback != NULL){
                k4a_playback_close(playback);
            }
            return -1;
                
        }

        k4a_calibration_intrinsic_parameters_t *intrinsics = &calibration.color_camera_calibration.intrinsics.parameters;

        // Stream the Playback to actually extract frames from it
        while(stream_result == K4A_STREAM_RESULT_SUCCEEDED){
        
            stream_result = k4a_playback_get_next_capture(playback,&capture);
            
            if (stream_result == K4A_STREAM_RESULT_EOF)
            {
                // End of file reached
                break;
            }
            
            bool flag = true;
            while(flag){ // loop until we get a valid image
                k4a_image_t temp_image = k4a_capture_get_color_image(capture);
                if( 0 == k4a_image_get_width_pixels(temp_image) ){
                    k4a_stream_result_t stream_result = k4a_playback_get_next_capture(playback,&capture);
                    if (stream_result == K4A_STREAM_RESULT_EOF)
                    {
                        std::cout << "ERROR: Recording file is empty: " << inputFname << std::endl;
                        throw std::runtime_error("Recording file is empty || EOF");
                    }
                    else if (stream_result == K4A_STREAM_RESULT_FAILED)
                    {
                        std::cout << "ERROR: Failed to read first capture from file: " << inputFname << std::endl;
                        throw std::runtime_error("Recording file is empty || FAILED");
                    
                    }
                }

                else{
                    std::cout << "Finally got a valid Image" << std::endl;
                    std::cout <<  "\n\n *************Color Image info: " << k4a_image_get_width_pixels(temp_image) << " -- " <<  k4a_image_get_height_pixels(temp_image) <<  " *************\n\n"<<std::endl;
                    flag = false;
                    break;
                }
            }
            k4a_image_t color_image = k4a_capture_get_color_image(capture);
            std::cout << "Image dimensions : " << k4a_image_get_height_pixels(color_image) << " , " << k4a_image_get_width_pixels(color_image) << std::endl;
            // Now we have a valid color image .. lets detect tags!
            float l = 0.3048f; // modify this to match the actual length of the box you are using
                                // our box is 12 inches -> 30.48cm


            //cv::Mat cv_image_with_alpha(im.get_height_pixels(), im.get_width_pixels(), CV_8UC4, (void *)im.get_buffer());
            //cv::Mat cv_image_no_alpha;
            //cv::cvtColor(cv_image_with_alpha, cv_image_no_alpha, cv::COLOR_BGRA2BGR);
            
            
            //const uint8_t* cv_color = reinterpret_cast<const uint8_t*>(k4a_image_get_buffer(color_image));
            //cv::Mat image( k4a_image_get_height_pixels(color_image), k4a_image_get_width_pixels(color_image), CV_8UC4, (void*)cv_color, cv::Mat::AUTO_STEP);
            cv::Mat image( k4a_image_get_height_pixels(color_image), k4a_image_get_width_pixels(color_image), CV_8UC4, (void*)k4a_image_get_buffer(color_image), cv::Mat::AUTO_STEP);
            std::cout <<  "Created color image OPENCV : " << image.size << std::endl;
            //cv::imwrite("rec1.jpg", image);
            
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
            std::cout <<  "Created gray image OPENCV" << std::endl;
            return 0;
            image_u8_t orig { .width = gray.cols,
                .height = gray.rows,
                .stride = gray.cols,
                .buf = gray.data
            };
            // cv::imshow("RGB Image", image);
            //cv::waitKey(0);
            TagDetector td_obj;
            zarray_t* detections = apriltag_detector_detect(td_obj.td, &orig);
            std::cout <<"Detected " << zarray_size(detections) << " fiducial markers" << std::endl;
            bool found_tag = false;

            for (int i = 0; i < zarray_size(detections); i++){
                apriltag_detection_t *det;
                zarray_get(detections, i, &det);
                
                std::cout <<"Camera "<< camID << " detected marker ID : " << det->id << std::endl;
                if (det->id > 5) {
                    std::cout <<"Camera "<< camID <<" detected incorrect marker #" << det->id << std::endl;;
                    continue;
                }

                apriltag_detection_info_t info;
                info.det = det;
                info.cx = intrinsics->param.cx; // pixels
                info.cy = intrinsics->param.cy;
                info.fx = intrinsics->param.fx; // mm
                info.fy = intrinsics->param.fy;
                info.tagsize = 0.14f; //  14cm in meters -- need to change this once i make measurements. 
                apriltag_pose_t pose;
                double err = estimate_tag_pose(&info, &pose);

                const double* tr = &pose.R->data[0];
                const double* tt = &pose.t->data[0];

                std::cout <<"Object-space error = " << err << std::endl;
                std::cout <<"R = [ " << tr[0] << ", " << tr[1] << ", " << tr[2] << std::endl;
                std::cout <<"      " << tr[3] << ", " << tr[4] << ", " << tr[5] << std::endl;
                std::cout <<"      " << tr[6] << ", " << tr[7] << ", " << tr[8] << " ]" << std::endl;
                std::cout <<"t = [ " << tt[0] << ", " << tt[1] << ", " << tt[2] << " ]" << std::endl;

                Eigen::Matrix4f transform;
                Eigen::Matrix3f rotmat;
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        transform(row, col) = static_cast<float>( tr[row * 3 + col] );
                        rotmat(row, col) = static_cast<float>( tr[row * 3 + col] );
                    }
                }
                for (int row = 0; row < 3; ++row) {
                    transform(row, 3) = static_cast<float>( tt[row] );
                }
                for (int col = 0; col < 3; ++col) {
                    transform(3, col) = 0.f;
                }
                transform(3, 3) = 1.f;
                Eigen::Matrix4f tag_pose = transform; 
                auto current_transform = tag_pose.cast<double>();
                std::cout << "Cam : " << camID << " found tag #" << det->id << " at: \n\t" << current_transform << std::endl;
                found_tag = true;
            }
            if (!found_tag) {
                std::cout  <<"Camera "<< camID << " did not observe the fiducial marker - Waiting for the next frame" << std::endl;
                //continue;
            }

            k4a_image_release(color_image);
            k4a_capture_release(capture);
        }

        if (playback != NULL)
        {
            k4a_playback_close(playback);
            returnCode = -1;
        }

        if (capture != NULL)
        {
            k4a_capture_release(capture);
            returnCode = -1;
        }

        if (stream_result == K4A_STREAM_RESULT_FAILED){
            std::cout<<"Failed to read entire recording" << std::endl;
            returnCode = -1;
        }
    }
    return returnCode;
}

        