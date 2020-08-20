package com.ml.quaterion.facenetdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

// Utility class for FaceNet mdoel
class FaceNetModel( context : Context ) {

    // TFLiteInterpreter used for running the FaceNet model.
    private var interpreter : Interpreter

    // Input image size for FaceNet model.
    private val imgSize = 160

    init {
        // Initialize TFLiteInterpreter
        val interpreterOptions = Interpreter.Options().apply {
            setNumThreads( 4 )
        }
        interpreter = Interpreter(FileUtil.loadMappedFile(context, "facenet_int8_quant.tflite") , interpreterOptions )
    }

    // Gets an face embedding using FaceNet
    fun getFaceEmbedding( image : Bitmap , crop : Rect , preRotate: Boolean ) : FloatArray {
        return runFaceNet(
            convertBitmapToBuffer(
                cropRectFromBitmap( image , crop , preRotate )
            )
        )[0]
    }

    // Run the FaceNet model.
    private fun runFaceNet(inputs: Any): Array<FloatArray> {
        val t1 = System.currentTimeMillis()
        val outputs = Array(1) { FloatArray(128 ) }
        interpreter.run(inputs, outputs)
        Log.i( "Performance" , "FaceNet Inference Speed in ms : ${System.currentTimeMillis() - t1}")
        return outputs
    }

    // Resize the given bitmap and convert it to a ByteBuffer
    private fun convertBitmapToBuffer( image : Bitmap) : ByteBuffer {
        val imageByteBuffer = ByteBuffer.allocateDirect( 1 * imgSize * imgSize * 3 * 4 )
        imageByteBuffer.order( ByteOrder.nativeOrder() )
        val resizedImage = Bitmap.createScaledBitmap(image, imgSize , imgSize, true)
        for (x in 0 until imgSize) {
            for (y in 0 until imgSize) {
                val pixelValue = resizedImage.getPixel( x , y )
                imageByteBuffer.putFloat((((pixelValue shr 16 and 0xFF) - 128f) / 128f))
                imageByteBuffer.putFloat((((pixelValue shr 8 and 0xFF) - 128f) / 128f ))
                imageByteBuffer.putFloat((((pixelValue and 0xFF) - 128f )/ 128f))
            }
        }
        return imageByteBuffer
    }

    // Crop the given bitmap with the given rect.
    private fun cropRectFromBitmap(source: Bitmap, rect: Rect , preRotate : Boolean ): Bitmap {
        val cropped = Bitmap.createBitmap(
            if ( preRotate ) rotateBitmap( source , 90f )!! else source,
            rect.left,
            rect.top,
            rect.width(),
            rect.height()
        )
        return cropped
    }

    private fun rotateBitmap(source: Bitmap, angle: Float): Bitmap? {
        val matrix = Matrix()
        matrix.postRotate( angle )
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix , false )
    }

}