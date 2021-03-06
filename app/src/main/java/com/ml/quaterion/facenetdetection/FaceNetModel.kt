/*
 * Copyright 2020 Shubham Panchal
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.ml.quaterion.facenetdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Rect
import android.os.Environment
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

// Utility class for FaceNet model
class FaceNetModel( context : Context ) {

    // TFLiteInterpreter used for running the FaceNet model.
    private var interpreter : Interpreter

    // Input image size for FaceNet model.
    private val imgSize = 112

    // Output embedding size
    private val embeddingDim = 192

    // Image Processor for preprocessing input images.
    private val imageTensorProcessor = ImageProcessor.Builder()
            .add( ResizeOp( imgSize , imgSize , ResizeOp.ResizeMethod.BILINEAR ) )
            .add( NormalizeOp( 127.5f , 127.5f ) )
            .build()

    init {
        // Initialize TFLiteInterpreter
        val interpreterOptions = Interpreter.Options().apply {
            setNumThreads( 4 )
        }
        interpreter = Interpreter(FileUtil.loadMappedFile(context, "facenet_float16.tflite") , interpreterOptions )
    }

    // Gets an face embedding using FaceNet, use the `crop` rect.
    fun getFaceEmbedding( image : Bitmap , crop : Rect , preRotate: Boolean ) : FloatArray {
        return runFaceNet(
            convertBitmapToBuffer(
                cropRectFromBitmap( image , crop , preRotate )
            )
        )[0]
    }

    // Gets an face embedding using the FaceNet model, given the cropped images.
    fun getFaceEmbeddingWithoutBBox( image : Bitmap ) : FloatArray {
        return runFaceNet( convertBitmapToBuffer( image ) )[0]
    }

    // Run the FaceNet model.
    private fun runFaceNet(inputs: Any): Array<FloatArray> {
        val t1 = System.currentTimeMillis()
        val outputs = Array(1) { FloatArray(embeddingDim ) }
        interpreter.run(inputs, outputs)
        Log.i( "Performance" , "FaceNet Inference Speed in ms : ${System.currentTimeMillis() - t1}")
        return outputs
    }

    // Resize the given bitmap and convert it to a ByteBuffer
    private fun convertBitmapToBuffer( image : Bitmap) : ByteBuffer {
        val imageTensor = imageTensorProcessor.process( TensorImage.fromBitmap( image ) )
        return imageTensor.buffer
    }

    // Crop the given bitmap with the given rect.
    private fun cropRectFromBitmap(source: Bitmap, rect: Rect , preRotate : Boolean ): Bitmap {
        var width = rect.width()
        var height = rect.height()
        if ( (rect.left + width) > source.width ){
            width = source.width - rect.left
        }
        if ( (rect.top + height ) > source.height ){
            height = source.height - rect.top
        }
        val croppedBitmap = Bitmap.createBitmap(
                if ( preRotate ) rotateBitmap( source )!! else source,
                rect.left,
                rect.top,
                width,
                height )
        //saveBitmap( croppedBitmap , "image")
        return croppedBitmap
    }

    private fun saveBitmap(image: Bitmap, name: String) {
        val fileOutputStream =
                FileOutputStream(File( Environment.getExternalStorageDirectory()!!.absolutePath + "/$name.png"))
        image.compress(Bitmap.CompressFormat.PNG, 100, fileOutputStream)
    }


    private fun rotateBitmap(source: Bitmap): Bitmap? {
        val matrix = Matrix()
        matrix.postRotate( -90f )
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix , false )
    }

}