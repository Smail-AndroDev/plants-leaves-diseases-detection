package com.inf.tflite;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.Manifest;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.inf.tflite.ml.AutoModel1;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.DecimalFormat;

public class MainActivity extends AppCompatActivity {
    private ImageView imgView;
    private ImageButton select, predict,capture;
    private TextView confidence,className;
    private Bitmap img;
    ImageProcessor imageProcessor;
    String[] classes={"Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"};
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageProcessor= new ImageProcessor.Builder().
                add(new ResizeOp(256,256, ResizeOp.ResizeMethod.BILINEAR))
                .build();

        imgView = (ImageView) findViewById(R.id.imageView);
        select = (ImageButton) findViewById(R.id.select);
        capture = (ImageButton) findViewById(R.id.capture);
//        predict = (Button) findViewById(R.id.predict);
        confidence= (TextView) findViewById(R.id.confidence);
        className= (TextView) findViewById(R.id.className);

        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 100);

            }
        });
//        predict.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//
//            }
//        });
        capture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                getPermission();
            }
        });
    }

    void predict(){
        try {

//                    img = Bitmap.createScaledBitmap(img, 256, 256, true);

            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(img);
            tensorImage=imageProcessor.process(tensorImage);

            AutoModel1 model = AutoModel1.newInstance(MainActivity.this);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 256, 256, 3}, DataType.FLOAT32);


            Log.d("shape", tensorImage.getBuffer().toString());
            Log.d("shape", inputFeature0.getBuffer().toString());

//                    inputFeature0.loadBuffer(TensorImage.fromBitmap(img).getBuffer());
            inputFeature0.loadBuffer(tensorImage.getBuffer());

            // Runs model inference and gets result.
            AutoModel1.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            int indx_max=findMax(outputFeature0.getFloatArray());
            Log.d("shape", "onClick: "+outputFeature0.getFloatArray()[0]);
            Log.d("shape", "onClick: "+outputFeature0.getFloatArray()[1]);
            Log.d("shape", "onClick: "+outputFeature0.getFloatArray()[2]);
            Log.d("shape", "max: "+indx_max);

            DecimalFormat df = new DecimalFormat("#.##");
            String roundedNumber = df.format(outputFeature0.getFloatArray()[indx_max]*100.0f);
            int green = Color.rgb(76, 175, 80);
            if(indx_max!=2){
                className.setTextColor(Color.RED);
            }else{
                className.setTextColor(green);

            }

            className.setText(classes[indx_max]);
            confidence.setText(""+roundedNumber+"%");
//            Toast.makeText(MainActivity.this, ""+classes[indx_max] , Toast.LENGTH_SHORT).show();
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == 100)
        {
            imgView.setImageURI(data.getData());

            Uri uri = data.getData();
            try {
                img = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                predict();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        if (requestCode == 300 && resultCode == RESULT_OK) {
            // Get the captured image as a Bitmap
            Bundle extras = data.getExtras();
            img = (Bitmap) extras.get("data");
            if (img != null) {
                // Display the captured image in ImageView
                imgView.setImageBitmap(img);
                predict();
            }
        }
    }

    public void getPermission(){
        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            // Request the camera permission
            ActivityCompat.requestPermissions(MainActivity.this,
                    new String[]{Manifest.permission.CAMERA},
                    200);
        } else {
            // Permission already granted, open camera
            openCamera();
        }

    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 200) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission granted, open camera
                openCamera();
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }
    private void openCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (cameraIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(cameraIntent, 300);
        } else {
            Toast.makeText(this, "Camera not available", Toast.LENGTH_SHORT).show();
        }
    }
    public static int findMax(float[] array) {
        // Check if the array is null or empty
        if (array == null || array.length == 0) {
            throw new IllegalArgumentException("Array is null or empty");
        }

        // Initialize max value with the first element of the array
        float max = 0;
        int indx_max=0;
        // Iterate through the array to find the maximum value
        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                indx_max=i;
            }
        }

        return indx_max;
    }
}