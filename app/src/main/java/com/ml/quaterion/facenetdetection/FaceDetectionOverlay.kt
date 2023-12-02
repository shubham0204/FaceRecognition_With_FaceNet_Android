package com.ml.quaterion.facenetdetection

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.widget.FrameLayout
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.graphics.toRectF
import androidx.core.view.doOnLayout
import androidx.lifecycle.LifecycleOwner
import com.google.common.util.concurrent.ListenableFuture
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

@SuppressLint("ViewConstructor")
@ExperimentalGetImage
class FaceDetectionOverlay(
    private val lifecycleOwner: LifecycleOwner ,
    private val cameraProviderFuture: ListenableFuture<ProcessCameraProvider> ,
    private val context: Context ,
    private val appViewModel: AppViewModel
) : FrameLayout( context ) {

    private var t1 = 0L
    private var overlayWidth: Int = 0
    private var overlayHeight: Int = 0
    private val realTimeOpts = FaceDetectorOptions.Builder()
        .setPerformanceMode( FaceDetectorOptions.PERFORMANCE_MODE_FAST )
        .build()
    private val detector = FaceDetection.getClient(realTimeOpts)

    private lateinit var imageTransform: Matrix
    private lateinit var boundingBoxTransform: Matrix
    private var isProcessing = false
    private val boundingBoxOverlay: BoundingBoxOverlay

    var predictions : Array<Prediction> = arrayOf()

    private fun isDetectorReady() : Boolean {
        return appViewModel.model.isInitialized && appViewModel.annotator.isInitialized
    }

    init {

        val previewView = PreviewView( context )
        val executor = ContextCompat.getMainExecutor( context )
        var frameAnalyzer: ImageAnalysis
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing( CameraSelector.LENS_FACING_BACK )
                .build()
            frameAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio( AspectRatio.RATIO_16_9 )
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            frameAnalyzer.setAnalyzer( Executors.newSingleThreadExecutor() , analyzer )
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview ,
                frameAnalyzer
            )
        }, executor )
        addView( previewView )

        val boundingBoxOverlayParams = LayoutParams(
            LayoutParams.MATCH_PARENT ,
            LayoutParams.MATCH_PARENT
        )
        boundingBoxOverlay = BoundingBoxOverlay( context )
        boundingBoxOverlay.setWillNotDraw( false )
        boundingBoxOverlay.setZOrderOnTop( true )
        addView( boundingBoxOverlay , boundingBoxOverlayParams )

        doOnLayout {
            overlayHeight = it.measuredHeight
            overlayWidth = it.measuredWidth

        }
    }


    private val analyzer = ImageAnalysis.Analyzer {image ->
        if( isProcessing ) {
            image.close()
            return@Analyzer
        }
        isProcessing = true
        // Rotated bitmap for the FaceNet model
        var frameBitmap = Bitmap.createBitmap( image.image!!.width , image.image!!.height , Bitmap.Config.ARGB_8888 )
        frameBitmap.copyPixelsFromBuffer( image.planes[0].buffer )

        println( "Image rotation => " + image.imageInfo.rotationDegrees.toFloat() )

        // Configure frameHeight and frameWidth for output2overlay transformation matrix.
        if( !this::imageTransform.isInitialized ) {
            imageTransform = Matrix()
            imageTransform.apply {
                postRotate( image.imageInfo.rotationDegrees.toFloat() )
                // if( appViewModel.cameraFacing.value == CameraSelector.LENS_FACING_FRONT ) {
                //    postScale( -1f , 1f , image.width.toFloat() , image.height.toFloat() )
                // }
            }
        }

        frameBitmap = Bitmap.createBitmap(
            frameBitmap , 0 , 0 , frameBitmap.width , frameBitmap.height ,
            imageTransform , false
        )

        if( !this::boundingBoxTransform.isInitialized ) {
            boundingBoxTransform = Matrix()
            boundingBoxTransform.apply {
                postScale(
                    overlayWidth / frameBitmap.width.toFloat() ,
                    overlayHeight / frameBitmap.height.toFloat()
                )
            }
        }

        val inputImage = InputImage.fromBitmap( frameBitmap , 0 )
        Log.e( "APP" , "frames received" )
        detector.process( inputImage )
            .addOnSuccessListener { faces ->
                CoroutineScope( Dispatchers.Default ).launch {
                    if( appViewModel.model.isInitialized && appViewModel.annotator.isInitialized ) {
                        runModel( faces , frameBitmap )
                    }
                    else {
                        showFaces( faces )
                    }
                }
            }
            .addOnCompleteListener {
                image.close()
            }
    }

    private fun validateRect(
        cameraFrameBitmap: Bitmap ,
        boundingBox: Rect
    ) : Boolean {
        return boundingBox.left >= 0 &&
                boundingBox.top >= 0 &&
                (boundingBox.left + boundingBox.width()) < cameraFrameBitmap.width &&
                (boundingBox.top + boundingBox.height()) < cameraFrameBitmap.height
    }


    private suspend fun runModel(faces : List<Face>, cameraFrameBitmap : Bitmap){
        withContext( Dispatchers.Default ) {
            t1 = System.currentTimeMillis()
            val predictions = ArrayList<Prediction>()
            faces.filter{ validateRect( cameraFrameBitmap , it.boundingBox ) }
                 .forEach {
                Log.e( "APP" , "Faces detected..." )
                val croppedBitmap = BitmapUtils.cropRectFromBitmap( cameraFrameBitmap , it.boundingBox )
                val label = appViewModel.annotator.value!!.run( appViewModel.model.value!!.getFaceEmbedding( croppedBitmap ) )
                cameraFrameBitmap.recycle()
                croppedBitmap.recycle()
                Logger.log( "Person identified as $label" )
                val box = it.boundingBox.toRectF()
                boundingBoxTransform.mapRect( box )
                predictions.add(
                    Prediction(
                        box,
                        label ,
                    )
                )
                Log.e( "Performance" , "Inference time -> ${System.currentTimeMillis() - t1}")
            }
            withContext( Dispatchers.Main ) {
                this@FaceDetectionOverlay.predictions = predictions.toTypedArray()
                boundingBoxOverlay.invalidate()
                isProcessing = false
            }
        }
    }

    private suspend fun showFaces( faces: List<Face> ) {
        withContext( Dispatchers.Default ) {
            Log.e( "APP" , "Detecting faces directly..." )
            val predictions = ArrayList<Prediction>()
            faces.forEach {
                val box = it.boundingBox.toRectF()
                boundingBoxTransform.mapRect( box )
                predictions.add(
                    Prediction(
                        box ,
                        "Face: ${it.trackingId}" ,
                    )
                )
            }
            withContext( Dispatchers.Main ) {
                this@FaceDetectionOverlay.predictions = predictions.toTypedArray()
                boundingBoxOverlay.invalidate()
                isProcessing = false
            }
        }
    }

    inner class BoundingBoxOverlay( context: Context )
        : SurfaceView( context ) , SurfaceHolder.Callback {

        // Paint for boxes and text
        private val boxPaint = Paint().apply {
            color = Color.parseColor("#4D90caf9")
            style = Paint.Style.FILL
        }
        private val textPaint = Paint().apply {
            strokeWidth = 2.0f
            textSize = 32f
            color = Color.WHITE
        }

        override fun surfaceCreated(holder: SurfaceHolder) {
            TODO("Not yet implemented")
        }

        override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
            TODO("Not yet implemented")
        }

        override fun surfaceDestroyed(holder: SurfaceHolder) {
            TODO("Not yet implemented")
        }

        override fun onDraw(canvas: Canvas) {
            Log.e( "APP" , "Drawing on view............" )
            predictions.forEach {
                Log.e( "APP" , "Drew faces on view............" )
                canvas.drawRoundRect(it.bbox, 16f, 16f, boxPaint)
                canvas.drawText(
                    it.label,
                    it.bbox.centerX(),
                    it.bbox.centerY(),
                    textPaint
                )
            }
        }

    }


}