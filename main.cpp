#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <QCoreApplication>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace cv;
using namespace std;

/** Functions to use */
void dni_find(Mat img_object, Mat frame);
Mat CorrectPerspective(Mat frame, vector<Point2f> scene_corners);
void detectFace(Mat croppedImage);
Mat getMRZ(Mat img);
Mat getSignature(Mat img);


/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
bool trackObject=false;
bool frontal=true;
bool nuevo=true;
Mat img_scene, faceROI;


/** @function main */
int main( int argc, char** argv )
{
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading 1\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading 2\n"); return -1; };

    Mat img_object;

    Mat img_object1 = imread( "Front_New.jpg", CV_LOAD_IMAGE_COLOR );
    //Mat img_object1 = imread( argv[1], CV_LOAD_IMAGE_COLOR );
    resize(img_object1,img_object1,Size(600,500));
    blur(img_object1,img_object1, Size(9,9));
    cvtColor(img_object1,img_object1, CV_BGR2GRAY);

    Mat img_object2 = imread( "Rear_New.jpg", CV_LOAD_IMAGE_COLOR );
    //Mat img_object2 = imread( argv[2], CV_LOAD_IMAGE_COLOR );
    resize(img_object2,img_object2,Size(600,500));
    blur(img_object2,img_object2, Size(11,11));
    cvtColor(img_object2,img_object2, CV_BGR2GRAY);

    Mat img_object3 = imread( "Front_Old.jpg", CV_LOAD_IMAGE_COLOR );
    //Mat img_object3 = imread( argv[3], CV_LOAD_IMAGE_COLOR );
    resize(img_object3,img_object3,Size(600,500));
    blur(img_object3,img_object3, Size(11,11));
    cvtColor(img_object3,img_object3, CV_BGR2GRAY);

    Mat img_object4 = imread( "Rear_Old.jpg", CV_LOAD_IMAGE_COLOR );
    //Mat img_object4 = imread( argv[4], CV_LOAD_IMAGE_COLOR );
    resize(img_object4,img_object4,Size(600,500));
    blur(img_object4,img_object4, Size(11,11));
    cvtColor(img_object4,img_object4, CV_BGR2GRAY);

    Mat frame;
    VideoCapture cap(0);
    if(!cap.isOpened())
        return -1;
    int n = 0;
    char filename[200];
    for(;;)
    {
        if(frontal==true && nuevo==true){
            img_object=img_object1; //Frontal and new DNI
        }
        if(frontal==false && nuevo==true){
            img_object=img_object2; //Rear and new DNI
        }
        if(frontal==true && nuevo==false){
            img_object=img_object3; //Frontal and old DNI
        }
        if(frontal==false && nuevo==false){
            img_object=img_object4; //Rear and old DNI
        }

        cap >> frame;
        //-- 3. Apply the classifier to the frame
        if( !frame.empty() )
        {
            dni_find(img_object, frame);
        }
        else
        { printf(" --(!) No captured frame -- Break!"); break; }

        char key = (char)waitKey(5); //delay N millis, usually long enough to display and capture input
        switch (key) {
        case 'a':
        case 'A':
            frontal=!frontal; //Change between frontal and back
            break;

        case 'b':
        case 'B':
            nuevo=!nuevo; //Change between new dni and old dni
            break;

        case 'q':
        case 'Q':
        case 27: //escape key
            return 0;

        case ' ': //Save an image
            sprintf(filename,"filename%.3d.jpg",n++);
            imwrite(filename,frame);
            cout << "Saved " << filename << endl;
            break;
        default:
            break;
        }
    }
    return 0;
    }


void dni_find(Mat img_object, Mat frame){

  Mat mostrar;
  frame.copyTo(mostrar);
  blur(frame,img_scene, Size(9,9));
  cvtColor(img_scene,img_scene, CV_BGR2GRAY);

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  SurfFeatureDetector detector( minHessian );

  std::vector<KeyPoint> keypoints_object, keypoints_scene;

  detector.detect( img_object, keypoints_object );
  detector.detect( img_scene, keypoints_scene );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_object, descriptors_scene;

  extractor.compute( img_object, keypoints_object, descriptors_object );
  extractor.compute( img_scene, keypoints_scene, descriptors_scene );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 400;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 3*min_dist )
     { good_matches.push_back( matches[i]); }
  }
  if(good_matches.size()<4){
      return;
  }

  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  Mat H = findHomography( obj, scene, CV_RANSAC );

  //-- Get the corners from the image_scene ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);

  //-- Draw lines between the corners (img_object + img_scene)
  line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

  //-- Show detected matches (img_object + img_scene)
  imshow( "A to change side, B to change version. Q or Escape to close the program", img_matches );

  //-- Draw lines between the corners (only scene)
  line( mostrar, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4 );
  line( mostrar, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 4 );
  line( mostrar, scene_corners[2], scene_corners[3], Scalar( 0, 255, 0), 4 );
  line( mostrar, scene_corners[3], scene_corners[0], Scalar( 0, 255, 0), 4 );

  //-- Show detected matches (only scene)
  imshow( "A to change side, B to change version, Q or Escape to close the program", mostrar );

  //-- Detect if the object is a rectangle

  //-- This calculates the length of one side
  float tamano=abs(scene_corners[1].x-scene_corners[0].x);
  //-- This calculates the difference between two sides
  float comparacion=(abs(scene_corners[1].x-scene_corners[0].x))-(abs(scene_corners[2].x-scene_corners[3].x));
  //-- This calculates the relation between width and height
  float relacion=(abs(scene_corners[3].y-scene_corners[0].y))/(abs(scene_corners[1].x-scene_corners[0].x));

  if(tamano > 10 && comparacion>-20 && comparacion<20 && relacion>0.65 && relacion<0.70){

      //-- These are the coordinates of the rectangle's corners. It isn't information that is needed to know,
      //   so it's comented, as well the info used to determinate the rectangle
    /*
      cout << "It is a rectangle" << endl;
      cout << "Coordinate X [0]" <<scene_corners[0].x << endl;
      cout << "Coordinate Y [0]" <<scene_corners[0].y << endl;
      cout << "Coordinate X [1]" <<scene_corners[1].x << endl;
      cout << "Coordinate Y [1]" <<scene_corners[1].y << endl;
      cout << "Coordinate X [2]" <<scene_corners[2].x << endl;
      cout << "Coordinate Y [2]" <<scene_corners[2].y << endl;
      cout << "Coordinate X [3]" <<scene_corners[3].x << endl;
      cout << "Coordinate Y [3]" <<scene_corners[3].y << endl;
      cout <<"Length: "<< tamano << endl;
      cout <<"Difference: "<< comparacion << endl;
      cout <<"Width/height: "<< relacion << endl;
    */
      //-- Crop the image
      Mat transformed=CorrectPerspective(frame, scene_corners);

      //-- If the image used to detect the dni is the front, we call the function to detect the face directly
      if (frontal==true){
      detectFace(transformed);
      getSignature(transformed);
      }
      //-- If the image used to detect the dni is the rear, we call the function to detect the MRZ directly
      else{
      getMRZ(transformed);
       }
  }
}

/** @function CorrectPerspective */
Mat CorrectPerspective(Mat frame, vector<Point2f> scene_corners)
{

    //-- Compute the size of the card by keeping the aspect ratio of an ID card
    double ratio=1.57;
    //-- Calculate the longest width and get the height from that width with the aspect ratio
    double cardW1=sqrt((scene_corners[3].x-scene_corners[2].x)*(scene_corners[3].x-scene_corners[2].x)+
            (scene_corners[3].y-scene_corners[2].y)*(scene_corners[3].y-scene_corners[2].y));
    double cardW2=sqrt((scene_corners[1].x-scene_corners[0].x)*(scene_corners[1].x-scene_corners[0].x)+
            (scene_corners[1].y-scene_corners[0].y)*(scene_corners[1].y-scene_corners[0].y));
    double cardW = max(cardW1, cardW2);
    double cardH=cardW/ratio;

    //-- Get the points of the image to show at the screen
    vector<Point2f> dst;

    dst.push_back(Point2f(0,0));
    dst.push_back(Point2f(cardW-1,0));
    dst.push_back(Point2f(cardW-1,cardH-1));
    dst.push_back(Point2f(0,cardH-1));

    //-- Get the perspective transform from ordered_points to dst and declare the output of warpPerspective
    Mat transmtx = getPerspectiveTransform(scene_corners,dst);
    Mat transformed = Mat::zeros(cardH, cardW, CV_8UC3);

    warpPerspective(frame, transformed, transmtx, transformed.size());

    imshow("Perspectiva", transformed);
    waitKey(0);
    return transformed;
}


///** @function detectFace */
 void detectFace(Mat croppedImage )
 {
     std::vector<Rect> faces;
     Mat frame_gray;
     Rect rect_face;

     if (1){//!trackObject){ //-- Detect faces and keep only the biggest one

        //-- We'll cut the face on a copy of the scene.
        Mat cortar=croppedImage.clone();
        cvtColor( croppedImage, frame_gray, CV_BGR2GRAY );
        face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
        //cout << "Faces.size: " << faces.size() << endl;

        int max_face_width = 0;
        int index;
        //-- We keep the biggest one
        if (faces.size()>0){
             trackObject = true;

             for( size_t i = 0; i < faces.size(); i++ )
             {
                 if (faces[i].width > max_face_width)
                 {
                     max_face_width = faces[i].width;
                     index = i;
                 }
             }

             rect_face = faces[index];
             //-- Draw a rectangle in the face section of the card
             rectangle( croppedImage,rect_face, Scalar( 255, 0, 255 ), 4, 4, 0 );
             faceROI = cortar( rect_face );

             //-- Show just the face and the detection on the scene
             imshow("face", faceROI);
             imshow("result face", croppedImage);
             waitKey(0);
         }
     }
 }

 ///** @function getSignature */
 Mat getSignature(Mat img){

     Mat RectElement = getStructuringElement(MORPH_RECT, Size(13, 5));
     Mat img_signature;
     Mat img_aux;
     vector<vector<Point> > cnt;
     resize(img,img_aux,Size(600,500));

     //-- Convert image to gray scale
     cvtColor(img_aux,img_signature, CV_BGR2GRAY);

     //-- Reduce high frequency noise by blurring the image
     GaussianBlur(img_signature,img_signature, Size(3, 3), 0);

     //-- blackhat operator is used to reveal dark regions against light backgrounds
     morphologyEx(img_signature,img_signature,MORPH_BLACKHAT, RectElement);
     imshow("Black hat", img_signature);
     //-- This closing operation is meant to close gaps in between the dni characters.
     morphologyEx(img_signature,img_signature, MORPH_CLOSE, RectElement);
    imshow("Closing", img_signature);
     //-- Threshold the image to find white regions
     threshold(img_signature,img_signature,0 ,255,THRESH_BINARY | THRESH_OTSU);
    imshow("Threshold", img_signature);
       // Find contours
     findContours( img_signature, cnt, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

     vector<Moments> mu(cnt.size());
      vector<Point2f> mc(cnt.size());
      vector<vector<Point> > contours_poly( cnt.size() );
      vector<RotatedRect> boundRect( cnt.size() );
      vector<Rect> mRect( cnt.size() );
      Point2f pt;
      float area_ref=0.0;
      float area;
      int index;

           for(int i=0;i<cnt.size();i++){
               mu[i]=moments(cnt[i],false);
           }

           for(int i=0;i<cnt.size();i++){
               mc[i]=Point2f(mu[i].m10/mu[i].m00,mu[i].m01/mu[i].m00); //Calculating the mass center
               circle(img_signature,mc[i],2,Scalar(255),2);
       if( mc[i].y>img_signature.rows/2 ){ // if the mass center is in the bottom half

           approxPolyDP( Mat(cnt[i]), contours_poly[i], 3, true );
            mRect[i] = boundingRect( Mat(contours_poly[i]) );

           boundRect[i] = minAreaRect( Mat(contours_poly[i]));
           pt=boundRect[i].size;

            int x=(int)pt.x;
             int y=(int)pt.y;
             area=x*y;
           if(area>area_ref){ // if it is the biggest area
               area_ref=area;
                index=i; //We take that rectangle
                cout<<pt.x<<"\n"<<pt.y<<"\n";
                          }
             }
         }
         //Cut the signature
           Mat signature = Mat(img_aux,mRect[index]);
           namedWindow( "signature",CV_WINDOW_AUTOSIZE );// Create a window for display.
           imshow("signature",signature);
           waitKey(0);

     return signature;
 }

///** @function getMRZ */
 Mat getMRZ(Mat img){

    Mat RectElement = getStructuringElement(MORPH_RECT, Size(13, 5));
    Mat Element = getStructuringElement(MORPH_RECT, Size(21, 21));
    Mat img_mrz;
    Mat img_aux;
    vector<vector<Point> > cnt;
    resize(img,img_aux,Size(600,500));

     //-- Convert image to gray scale
    cvtColor(img_aux,img_mrz, CV_BGR2GRAY);

    //-- Reduce high frequency noise by blurring the image
    GaussianBlur(img_mrz,img_mrz, Size(3, 3), 0);

    //-- Blackhat operator is used to reveal dark regions against light backgrounds
    morphologyEx(img_mrz,img_mrz,MORPH_BLACKHAT, RectElement);

    //-- This closing operation is meant to close gaps in between MRZ characters.
    morphologyEx(img_mrz,img_mrz, MORPH_CLOSE, RectElement);

    //-- Threshold the image to find white regions
    threshold(img_mrz,img_mrz,0 ,255,THRESH_BINARY | THRESH_OTSU);

    //-- This closing operation is meant to close gaps in between MRZ characters.
    morphologyEx(img_mrz,img_mrz, MORPH_CLOSE, Element);

    //-- Find contours
    findContours( img_mrz, cnt, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    vector<Moments> mu(cnt.size());
    vector<Point2f> mc(cnt.size());
    vector<vector<Point> > contours_poly( cnt.size() );
    vector<RotatedRect> boundRect( cnt.size() );
    vector<Rect> mRect( cnt.size() );
    Point2f pt;
    float area_ref=0.0;
    float area;
    int index;

    for(int i=0;i<cnt.size();i++){
       mu[i]=moments(cnt[i],false);
    }

    for(int i=0;i<cnt.size();i++){
       mc[i]=Point2f(mu[i].m10/mu[i].m00,mu[i].m01/mu[i].m00); //Calculating the mass center
       circle(img_mrz,mc[i],2,Scalar(255),2);
       if( mc[i].y>img_mrz.rows/2 ){ // if the mass center is in the bottom half

            approxPolyDP( Mat(cnt[i]), contours_poly[i], 3, true );
            mRect[i] = boundingRect( Mat(contours_poly[i]) );
            boundRect[i] = minAreaRect( Mat(contours_poly[i]));
            pt=boundRect[i].size;
            int x=(int)pt.x;
            int y=(int)pt.y;
            area=x*y;

            if(area>area_ref){ // if it is the biggest area
                area_ref=area;
                index=i; //We take that rectangle
                //cout<<pt.x<<"\n"<<pt.y<<"\n";
            }
        }

    }

    //-- Cut the MRZ
    Mat mrz = Mat(img_aux,mRect[index]);
    imshow("MRZ",mrz);
    waitKey(0);
    return mrz;
 }
