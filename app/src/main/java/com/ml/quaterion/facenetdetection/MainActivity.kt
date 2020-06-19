package com.ml.quaterion.facenetdetection

import android.Manifest
import android.app.ProgressDialog
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.gms.tasks.OnSuccessListener
import com.google.firebase.ml.vision.FirebaseVision
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.common.FirebaseVisionImageMetadata
import com.google.firebase.ml.vision.face.FirebaseVisionFace
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions
import java.io.*
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {

    private val REQUEST_CODE_PERMISSIONS = 10
    private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA , Manifest.permission.WRITE_EXTERNAL_STORAGE )
    private lateinit var cameraTextureView : TextureView
    private lateinit var frameAnalyser  : FrameAnalyser

    // For testing purposes only!
    companion object {
        // This view's VISIBILITY is set to View.GONE in activity_main.xml
        lateinit var logTextView : TextView
        fun setMessage ( message : String ) {
            logTextView.text = message
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Implementation of CameraX preview
        
        cameraTextureView = findViewById( R.id.camera_textureView )
        if (allPermissionsGranted()) {
            cameraTextureView.post { startCamera() }
        }
        else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }
        cameraTextureView.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
            updateTransform()
        }
        
        val boundingBoxOverlay = findViewById<BoundingBoxOverlay>( R.id.bbox_overlay )

        //This view's VISIBILITY is set to View.GONE in activity_main.xml
        logTextView = findViewById( R.id.logTextView )

        // Necessary to keep the Overlay above the TextureView so that the boxes are visible.
        boundingBoxOverlay.setWillNotDraw( false )
        boundingBoxOverlay.setZOrderOnTop( true )
        frameAnalyser = FrameAnalyser( this , boundingBoxOverlay)

        // Read image data
        scanStorageForImages()

    }

    private fun scanStorageForImages() {
        // Initialize FaceNet model.
        val model = FaceNetModel( this )

        // Create an empty ( String , FloatArray ) Hashmap for storing the data.
        val imageData = HashMap<String,FloatArray>()

        // Initialize Firebase MLKit Face Detector
        val accurateOps = FirebaseVisionFaceDetectorOptions.Builder()
            .setPerformanceMode(FirebaseVisionFaceDetectorOptions.ACCURATE)
            .build()
        val detector = FirebaseVision.getInstance().getVisionFaceDetector(accurateOps)

        if ( ContextCompat.checkSelfPermission( this , Manifest.permission.WRITE_EXTERNAL_STORAGE ) ==
            PackageManager.PERMISSION_GRANTED ) {
            val progressDialog = ProgressDialog( this )
            progressDialog.setMessage( "Loading images ..." )
            progressDialog.show()

            val imagesDir = File( Environment.getExternalStorageDirectory().absolutePath + "/images" )
            val imageSubDirs = imagesDir.listFiles()

            val subDirNames = imageSubDirs.map { file -> file.name }
            val subjectImages = imageSubDirs.map { file -> BitmapFactory.decodeFile( file.listFiles()[0].absolutePath ) }
            var imageCounter = 0
            val successListener = OnSuccessListener<List<FirebaseVisionFace?>> { faces ->
                if ( faces.isNotEmpty() ) {
                    imageData[ subDirNames[ imageCounter ] ] =
                        model.getFaceEmbedding( subjectImages[ imageCounter ] , faces[0]!!.boundingBox , false )
                    imageCounter += 1
                    // Make sure the frameAnalyser uses the given data!
                    frameAnalyser.faceList = imageData
                    if ( imageCounter == imageSubDirs.size ) {
                        progressDialog.dismiss()
                    }
                }
            }
            for ( image in subjectImages ) {
                val metadata = FirebaseVisionImageMetadata.Builder()
                    .setWidth( image.width )
                    .setHeight( image.height )
                    .setFormat(FirebaseVisionImageMetadata.IMAGE_FORMAT_NV21 )
                    .setRotation( FirebaseVisionImageMetadata.ROTATION_0 )
                    .build()
                val inputImage = FirebaseVisionImage.fromByteArray( bitmapToNV21( image ) , metadata )
                detector.detectInImage( inputImage ).addOnSuccessListener( successListener )
            }
        }




    }

    // Start the camera preview once the permissions are granted.
    private fun startCamera() {
        val previewConfig = PreviewConfig.Builder().apply {
            setTargetResolution(Size(640, 480 ))
            setLensFacing( CameraX.LensFacing.BACK )
        }.build()
        val preview = Preview(previewConfig)
        preview.setOnPreviewOutputUpdateListener {
            val parent = cameraTextureView.parent as ViewGroup
            parent.removeView( cameraTextureView )
            parent.addView( cameraTextureView , 0)
            cameraTextureView.surfaceTexture = it.surfaceTexture
            updateTransform()
        }

        // FrameAnalyser -> fetches camera frames and makes them in the analyse() method.
        val analyzerConfig = ImageAnalysisConfig.Builder().apply {
            setImageReaderMode(
                ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
            setLensFacing( CameraX.LensFacing.BACK )

        }.build()
        val analyzerUseCase = ImageAnalysis(analyzerConfig).apply {
            setAnalyzer( Executors.newSingleThreadExecutor() , frameAnalyser )
        }

        // Bind the preview and frameAnalyser.
        CameraX.bindToLifecycle(this, preview, analyzerUseCase )
    }

    private fun updateTransform() {
        val matrix = Matrix()
        val centerX = cameraTextureView.width.div(2f)
        val centerY = cameraTextureView.height.div(2f)
        val rotationDegrees = when(cameraTextureView.display.rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> return
        }
        matrix.postRotate(-rotationDegrees.toFloat(), centerX , centerY )
        cameraTextureView.setTransform(matrix)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                cameraTextureView.post { startCamera() }
                scanStorageForImages()
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission( baseContext, it ) == PackageManager.PERMISSION_GRANTED
    }

    private fun bitmapToNV21(bitmap: Bitmap): ByteArray {
        val argb = IntArray(bitmap.width * bitmap.height )
        bitmap.getPixels(argb, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val yuv = ByteArray(bitmap.height * bitmap.width + 2 * Math.ceil(bitmap.height / 2.0).toInt()
                * Math.ceil(bitmap.width / 2.0).toInt())
        encodeYUV420SP( yuv, argb, bitmap.width, bitmap.height)
        return yuv
    }

    private fun encodeYUV420SP(yuv420sp: ByteArray, argb: IntArray, width: Int, height: Int) {
        val frameSize = width * height
        var yIndex = 0
        var uvIndex = frameSize
        var R: Int
        var G: Int
        var B: Int
        var Y: Int
        var U: Int
        var V: Int
        var index = 0
        for (j in 0 until height) {
            for (i in 0 until width) {
                R = argb[index] and 0xff0000 shr 16
                G = argb[index] and 0xff00 shr 8
                B = argb[index] and 0xff shr 0
                Y = (66 * R + 129 * G + 25 * B + 128 shr 8) + 16
                U = (-38 * R - 74 * G + 112 * B + 128 shr 8) + 128
                V = (112 * R - 94 * G - 18 * B + 128 shr 8) + 128
                yuv420sp[yIndex++] = (if (Y < 0) 0 else if (Y > 255) 255 else Y).toByte()
                if (j % 2 == 0 && index % 2 == 0) {
                    yuv420sp[uvIndex++] = (if (V < 0) 0 else if (V > 255) 255 else V).toByte()
                    yuv420sp[uvIndex++] = (if (U < 0) 0 else if (U > 255) 255 else U).toByte()
                }
                index++
            }
        }
    }

}
