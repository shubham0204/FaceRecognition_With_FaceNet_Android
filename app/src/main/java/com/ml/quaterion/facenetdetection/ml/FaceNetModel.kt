/*
 * Copyright 2023 Shubham Panchal
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
package com.ml.quaterion.facenetdetection.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.ml.quaterion.facenetdetection.Logger
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

// Utility class for FaceNet model
class FaceNetModel( context : Context ,
                    var model : ModelInfo ,
                    useGpu : Boolean ,
                    useXNNPack : Boolean) {

    // Input image size for FaceNet model.
    private val imgSize = model.inputDims

    // Output embedding size
    val embeddingDim = model.outputDims
    private val faceNetModelOutputs = Array( 1 ){ FloatArray( embeddingDim ) }

    private var interpreter : Interpreter
    private val imageTensorProcessor = ImageProcessor.Builder()
        .add( ResizeOp( imgSize , imgSize , ResizeOp.ResizeMethod.BILINEAR ) )
        .add( StandardizeOp() )
        .build()

    companion object {
        init {
            System.loadLibrary( "facedetector" )
        }
    }

    init {
        // Initialize TFLiteInterpreter
        val interpreterOptions = Interpreter.Options().apply {
            // Add the GPU Delegate if supported.
            // See -> https://www.tensorflow.org/lite/performance/gpu#android
            if ( useGpu ) {
                if ( CompatibilityList().isDelegateSupportedOnThisDevice ) {
                    addDelegate( GpuDelegate( CompatibilityList().bestOptionsForThisDevice ))
                }
            }
            else {
                // Number of threads for computation
                numThreads = 4
            }
            setUseXNNPACK( useXNNPack )
            useNNAPI = true
        }
        interpreter = Interpreter(FileUtil.loadMappedFile(context, model.assetsFilename ) , interpreterOptions )
        Logger.log("Using ${model.name} model.")
    }


    // Gets an face embedding using FaceNet.
    fun getFaceEmbedding( image : Bitmap ) : FloatArray {
        return runFaceNet( convertBitmapToBuffer( image ))[0]
    }

    /*
    fun getFaceEmbeddingV2( image: Bitmap ) : FloatArray {
        val resizedImage = Bitmap.createScaledBitmap( image , this.imgSize , this.imgSize , true )
        val resizedImageBytes = IntBuffer.allocate( this.imgSize * this.imgSize * 3 )
        resizedImage.copyPixelsToBuffer( resizedImageBytes )
        val pixels = IntArray( this.imgSize * this.imgSize * 3 )
        resizedImageBytes.get( pixels )
        for( i in pixels.indices){
            pixels[ i ] =
        }
        resizedImageBytes.put( this.standardize( pixels ) )
    }*/


    // Run the FaceNet model.
    private fun runFaceNet(inputs: Any): Array<FloatArray> {
        val t1 = System.currentTimeMillis()
        interpreter.run( inputs, faceNetModelOutputs )
        Log.i( "Performance" , "${model.name} Inference Speed in ms : ${System.currentTimeMillis() - t1}")
        return faceNetModelOutputs
    }


    // Resize the given bitmap and convert it to a ByteBuffer
    private fun convertBitmapToBuffer( image : Bitmap ) : ByteBuffer {
        return imageTensorProcessor.process( TensorImage.fromBitmap( image ) ).buffer
    }

    // Op to perform standardization
    // x' = ( x - mean ) / std_dev
    class StandardizeOp : TensorOperator {

        external fun standardize(
            values: FloatArray
        ) : FloatArray

        override fun apply(p0: TensorBuffer?): TensorBuffer {
            val outputs = TensorBuffer.createFrom( p0!! , DataType.FLOAT32 )
            outputs.loadArray( standardize( p0.floatArray ) )
            return outputs
        }

    }

}
