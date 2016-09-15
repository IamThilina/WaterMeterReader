package com.imperialsoupgmail.tesseractexample;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.AppCompatSeekBar;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class EdgeDetector extends AppCompatActivity implements SeekBar.OnSeekBarChangeListener {

    private ImageView edgedetected_image_view;
    private Bitmap originalImage;
    private AppCompatSeekBar sb_threshold;
    private static final int ratio = 3;
    private TextView thresholdText;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_edge_detector);
        originalImage = BitmapFactory.decodeResource(getResources(),R.drawable.clear_far);
        sb_threshold = (AppCompatSeekBar)findViewById(R.id.sb_threshold);
        sb_threshold.setOnSeekBarChangeListener(this);
        edgedetected_image_view = (ImageView)findViewById(R.id.edgedetected_image);
        edgedetected_image_view.setImageBitmap(originalImage);
        thresholdText = (TextView) findViewById(R.id.text_threshold);
    }

    private Bitmap detectEdges(Bitmap bitmap, int threshold) {
        thresholdText.setText(String.valueOf(100+threshold));
        Mat rgba = new Mat(bitmap.getHeight(),bitmap.getWidth(), CvType.CV_8UC1);
        Utils.bitmapToMat(bitmap, rgba);
        Imgproc.resize(rgba, rgba, new Size(512, 309));

        Mat edges = new Mat(rgba.size(), CvType.CV_8UC1);
        Imgproc.cvtColor(rgba, edges, Imgproc.COLOR_RGB2GRAY, 4);
        Imgproc.Canny(edges, edges, threshold+100, (threshold+100)*ratio);

        Bitmap resultBitmap = Bitmap.createBitmap(edges.cols(), edges.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(edges, resultBitmap);
        return resultBitmap;
    }

    @Override
    public void onProgressChanged(SeekBar seekBar, int progress, boolean b) {
        Bitmap edged_detected_img = detectEdges(originalImage, progress);
        edgedetected_image_view.setImageBitmap(edged_detected_img);
    }

    @Override
    public void onStartTrackingTouch(SeekBar seekBar) {

    }

    @Override
    public void onStopTrackingTouch(SeekBar seekBar) {

    }
}
