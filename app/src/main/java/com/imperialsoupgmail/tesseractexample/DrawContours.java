package com.imperialsoupgmail.tesseractexample;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.AppCompatSeekBar;
import android.util.Log;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class DrawContours extends AppCompatActivity implements SeekBar.OnSeekBarChangeListener {

    private static final String TAG = "DrawContoursActivity";

    private BaseLoaderCallback mOpenCVCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    // Create and set View
                    //setContentView(R.layout.main);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    private ImageView contour_image_view;
    private Bitmap originalImage;
    private static final int ratio = 2;
    private static int threshold = 100;
    private AppCompatSeekBar sb_threshold;
    private TextView thresholdText;
    private TextView extractedText;
    private TessBaseAPI mTess; //Tess API reference
    String datapath = ""; //path to folder containing language data file

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_draw_contours);
        Log.i(TAG, "Trying to load OpenCV library");
        if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_2, this, mOpenCVCallBack))
        {
            Log.e(TAG, "Cannot connect to OpenCV Manager");
        } else{
            Log.i(TAG, "Connected to OpenCV Manager");
        }
        sb_threshold = (AppCompatSeekBar)findViewById(R.id.sb_edge_threshold);
        thresholdText = (TextView) findViewById(R.id.text_edge_threshold);
        extractedText = (TextView) findViewById(R.id.text_extracted_text);
        sb_threshold.setOnSeekBarChangeListener(this);
        originalImage = BitmapFactory.decodeResource(getResources(),R.drawable.clear_far);
        contour_image_view = (ImageView)findViewById(R.id.countour_image);
        contour_image_view.setImageBitmap(originalImage);

        datapath = getFilesDir()+ "/tesseract/";
        //make sure training data has been copied
        checkFile(new File(datapath + "tessdata/"));
        //init Tesseract API
        String language = "eng";
        mTess = new TessBaseAPI();
        mTess.init(datapath, language);
        mTess.setVariable(TessBaseAPI.VAR_CHAR_WHITELIST, "0123456789");
        mTess.setVariable(TessBaseAPI.VAR_CHAR_BLACKLIST,"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmopqrstuvwxyz");
        drawContours(originalImage);
    }

    private void copyFiles() {
        try {
            //location we want the file to be at
            String filepath = datapath + "tessdata/eng.traineddata";
            Log.d("DrawContoursActivity", filepath);
            //get access to AssetManager
            AssetManager assetManager = getAssets();

            //open byte streams for reading/writing
            InputStream instream = assetManager.open("tessdata/eng.traineddata");
            OutputStream outstream = new FileOutputStream(filepath);

            //copy the file to the location specified by filepath
            byte[] buffer = new byte[1024];
            int read;
            while ((read = instream.read(buffer)) != -1) {
                outstream.write(buffer, 0, read);
            }
            outstream.flush();
            outstream.close();
            instream.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void checkFile(File dir) {
        //directory does not exist, but we can successfully create it
        if (!dir.exists()&& dir.mkdirs()){
            copyFiles();
        }
        //The directory exists, but there is no data file in it
        if(dir.exists()) {
            String datafilepath = datapath+ "tessdata/eng.traineddata";
            File datafile = new File(datafilepath);
            if (!datafile.exists()) {
                copyFiles();
            }
        }
    }

    private void drawContours(Bitmap bitmap) {
        thresholdText.setText(String.valueOf(threshold));
        Mat rgba = new Mat(bitmap.getHeight(),bitmap.getWidth(), CvType.CV_8UC1);
        Utils.bitmapToMat(bitmap, rgba);
        Imgproc.resize(rgba, rgba, new Size(512, 384));

        Mat edges = new Mat(rgba.size(), CvType.CV_8UC1);

        // convert to gray scale
        Imgproc.cvtColor(rgba, edges, Imgproc.COLOR_RGB2GRAY, 4);
        // blurring
        //Imgproc.blur( edges, edges, new Size(3,3) );

        // detect edges using canny algorithm
        Imgproc.Canny(edges, edges, threshold, threshold*ratio);

        // rotate
        Mat rotate_transform = Imgproc.getRotationMatrix2D(new Point(edges.cols()/2, edges.rows()/2), -5, 1);
        //Imgproc.warpAffine(rgba,rgba,rotate_transform, rgba.size());

        // find contours
        Mat edgeCopy = edges.clone();
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edgeCopy, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // isolate the digits
        List<MatOfPoint> filteredContours = new ArrayList<MatOfPoint>();
        List<Rect> boundingRects = new ArrayList<>();
        Scalar color = new Scalar(0, 255, 0);

        // filter the contours based on their bounding rectangle
        for ( int contourIdx=0; contourIdx < contours.size(); contourIdx++ )
        {
            Rect boundRect = Imgproc.boundingRect(contours.get(contourIdx));
            if(boundRect.width > 5 && boundRect.width < 30 && boundRect.height < 30 && boundRect.height >= boundRect.width*1.7)  // Minimum size allowed for consideration
            {
                filteredContours.add(contours.get(contourIdx));
                boundingRects.add(boundRect);
            }
            /*if(boundRect.width > 10 && boundRect.width < 90 && boundRect.height < 90 && boundRect.height > 10)  // Minimum size allowed for consideration
            {
                filteredContours.add(contours.get(contourIdx));
                boundingRects.add(boundRect);
            }*/
        }

        // extract the maximum aligned contours
        List<MatOfPoint> digitContours = new ArrayList<MatOfPoint>();
        List<MatOfPoint> tempContours = new ArrayList<MatOfPoint>();
        Rect referenceRect, targetRect;
        int height;
        for ( int i=0; i < filteredContours.size(); i++ )
        {
            tempContours.clear();
            tempContours.add(filteredContours.get(i));
            referenceRect = boundingRects.get(i);
            height = referenceRect.height;
            for ( int j=0; j < filteredContours.size(); j++ )
            {
                targetRect = boundingRects.get(j);
                if(i != j){
                    if(((targetRect.tl().y >= referenceRect.tl().y) && (targetRect.tl().y <= referenceRect.tl().y+height)) || ((targetRect.br().y >= referenceRect.tl().y) && (targetRect.br().y <= referenceRect.tl().y+height))){
                        tempContours.add(filteredContours.get(j));
                    }

                }
            }
            if(tempContours.size() > digitContours.size())
                digitContours = tempContours;
        }

        // draw contours of filtered out digits
        //digitContours.clear();
        //digitContours = filteredContours;
        String recognizedText = "";
        Mat crop;
        Bitmap croppedBitMap;
        extractedText.setText(String.valueOf(digitContours.size()));
        ViewGroup layout = (ViewGroup) findViewById(R.id.filtered_rect_view);
        layout.removeAllViews();
        for ( int contourIdx=0; contourIdx < digitContours.size(); contourIdx++ )
        {
            Imgproc.drawContours(rgba, digitContours, contourIdx, color, 1);
            /*Rect boundRect = Imgproc.boundingRect(digitContours.get(contourIdx));
            Imgproc.rectangle(edges, boundRect.br(),boundRect.tl(), color, 1);
            crop = new Mat(edges, boundRect);
            Imgproc.resize(crop, crop, new Size(crop.width()*8, crop.height()*8));
            croppedBitMap = Bitmap.createBitmap(crop.cols(), crop.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(crop, croppedBitMap);
            ImageView imgView = new ImageView(this);
            imgView.setImageBitmap(croppedBitMap);
            layout.addView(imgView);*/
            //Imgproc.drawContours(rgba, digitContours, contourIdx, color, 1);
            Rect boundRect = Imgproc.boundingRect(digitContours.get(contourIdx));
            //Imgproc.rectangle(rgba, boundRect.br(),boundRect.tl(), color, 1);
            crop = new Mat(edges, boundRect);
            Imgproc.resize(crop, crop, new Size(crop.width()*6, crop.height()*6));
            croppedBitMap = Bitmap.createBitmap(crop.cols(), crop.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(crop, croppedBitMap);
            ImageView imgView = new ImageView(this);
            imgView.setImageBitmap(croppedBitMap);
            layout.addView(imgView);
            mTess.setImage(croppedBitMap);
            recognizedText += mTess.getUTF8Text();
        }


        //Imgproc.resize(rgba, rgba, new Size(rgba.width()*2, rgba.height()*2));
        Bitmap resultBitmap = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rgba, resultBitmap);
        contour_image_view.setImageBitmap(resultBitmap);

        extractedText.setText(String.valueOf(recognizedText));
    }

    @Override
    public void onProgressChanged(SeekBar seekBar, int progress, boolean b) {
        threshold = progress +100;
        drawContours(originalImage);
    }

    @Override
    public void onStartTrackingTouch(SeekBar seekBar) {

    }

    @Override
    public void onStopTrackingTouch(SeekBar seekBar) {

    }
}
