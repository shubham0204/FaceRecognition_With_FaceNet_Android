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
import android.graphics.*
import android.media.Image
import android.os.Environment
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.math.pow
import kotlin.math.sqrt

// Analyser class to process frames and produce detections.
class FrameAnalyser( private var context: Context , private var boundingBoxOverlay: BoundingBoxOverlay ) : ImageAnalysis.Analyzer {

    // Configure the FirebaseVisionFaceDetector
    private val realTimeOpts = FaceDetectorOptions.Builder()
            .setPerformanceMode( FaceDetectorOptions.PERFORMANCE_MODE_FAST )
            .build()
    private val detector = FaceDetection.getClient(realTimeOpts)

    // Used to determine whether the incoming frame should be dropped or processed.
    private var isProcessing = AtomicBoolean(false)

    // Store the face embeddings in a ( String , FloatArray ) ArrayList.
    // Where String -> name of the person and FloatArray -> Embedding of the face.
    var faceList = ArrayList<Pair<String,FloatArray>>()

    // FaceNet model utility class
    private val model = FaceNetModel( context )

    // Use any one of the two metrics, "cosine" or "l2"
    private val metricToBeUsed = "l2"

    // Here's where we receive our frames.
    override fun analyze(image: ImageProxy?, rotationDegrees: Int) {

        // android.media.Image -> android.graphics.Bitmap
        val bitmap = toBitmap( image?.image!! )

        // If the previous frame is still being processed, then skip this frame
        if (isProcessing.get()) {
            return
        }
        else {
            // Declare that the current frame is being processed.
            isProcessing.set(true)

            // Perform face detection
            val inputImage = InputImage.fromByteArray( BitmaptoNv21( bitmap )
                    , 640
                    , 480
                    , rotationDegrees
                    , InputImage.IMAGE_FORMAT_NV21
            )
            detector.process(inputImage)
                .addOnSuccessListener { faces ->
                    // Start a new thread to avoid frequent lags.
                    CoroutineScope( Dispatchers.Main ).launch {
                        runModel( faces , bitmap )
                    }
                }
                .addOnFailureListener { e ->
                    Log.e("Model", e.message)
                }
        }
    }

    private suspend fun runModel( faces : List<Face> , cameraFrameBitmap : Bitmap ){
        withContext( Dispatchers.Default ) {
            val predictions = ArrayList<Prediction>()
            for (face in faces) {
                try {
                    // Crop the frame using face.boundingBox.
                    // Convert the cropped Bitmap to a ByteBuffer.
                    // Finally, feed the ByteBuffer to the FaceNet model.
                    val subject = model.getFaceEmbedding(
                            cameraFrameBitmap ,
                            face.boundingBox ,
                            true ,
                            MainActivity.isRearCameraOn
                             )
                    Log.i( "Model" , "New frame received.")

                    // Perform clustering ( grouping )
                    // Store the clusters in a HashMap. Here, the key would represent the 'name'
                    // of that cluster and ArrayList<Float> would represent the collection of all
                    // L2 norms/ cosine distances.
                    val nameScoreHashmap = HashMap<String,ArrayList<Float>>()
                    for ( i in 0 until faceList.size ) {
                        // If this cluster ( i.e an ArrayList with a specific key ) does not exist,
                        // initialize a new one.
                        if ( nameScoreHashmap[ faceList[ i ].first ] == null ) {
                            // Compute the L2 norm and then append it to the ArrayList.
                            val p = ArrayList<Float>()
                            if ( metricToBeUsed == "cosine" ) {
                                Log.i( "Model" , "Using cosine similarity." )
                                p.add( cosineSimilarity( subject , faceList[ i ].second ) )
                            }
                            else {
                                Log.i( "Model" , "Using L2 norm." )
                                p.add( L2Norm(
                                        normalizeVector( subject ) ,
                                        normalizeVector( faceList[ i ].second )
                                ) )
                            }
                            nameScoreHashmap[ faceList[ i ].first ] = p
                        }
                        // If this cluster exists, append the L2 norm to it.
                        else {
                            if ( metricToBeUsed == "cosine" ) {
                                Log.i( "Model" , "Using cosine similarity." )
                                nameScoreHashmap[ faceList[ i ].first ]?.add( cosineSimilarity( subject , faceList[ i ].second ) )
                            }
                            else {
                                Log.i( "Model" , "Using L2 norm." )
                                nameScoreHashmap[ faceList[ i ].first ]?.add( L2Norm( subject , faceList[ i ].second ) )
                            }
                        }
                    }

                    // Compute the average of all scores norms for each cluster.
                    val avgScores = nameScoreHashmap.values.map{ scores ->
                        scores.toFloatArray().average()
                    }
                    Log.i( "Model" , "Average score for each user : $nameScoreHashmap" )
                    // Get the names of unique users
                    val names = nameScoreHashmap.keys.map{ key -> key }

                    // Calculate the minimum L2 distance from the stored average L2 norms.
                    var bestScoreUserName : String
                    if ( metricToBeUsed == "cosine" ) {
                        // In case of cosine similarity, choose the highest value.
                        bestScoreUserName = names[ avgScores.indexOf( avgScores.max()!! ) ]
                    }
                    else {
                        // In case of L2 norm, choose the lowest value.
                        bestScoreUserName = names[ avgScores.indexOf( avgScores.min()!! ) ]
                    }

                    Log.i( "Model" , "Person identified as ${bestScoreUserName}" )
                    // Push the results in form of a Prediction.
                    predictions.add(
                            Prediction(
                                    face.boundingBox,
                                    bestScoreUserName
                            )
                    )
                }
                catch ( e : Exception ) {
                    // If any exception occurs with this box and continue with the next boxes.
                    Log.e( "Model" , "Exception in FrameAnalyser : ${e.message}" )
                    continue
                }
            }
            withContext( Dispatchers.Main ) {
                // Clear the BoundingBoxOverlay and set the new results ( boxes ) to be displayed.
                boundingBoxOverlay.faceBoundingBoxes = predictions
                boundingBoxOverlay.invalidate()

                // Declare that the processing has been finished and the system is ready for the next frame.
                isProcessing.set(false)
            }
        }
    }


    // Use this method to save a Bitmap to the internal storage of your device.
    private fun saveBitmap(image: Bitmap, name: String) {
        val fileOutputStream =
            FileOutputStream(File( Environment.getExternalStorageDirectory()!!.absolutePath + "/$name.png"))
        image.compress(Bitmap.CompressFormat.PNG, 100, fileOutputStream)
    }

    // Compute the L2 norm of ( x2 - x1 )
    private fun L2Norm(x1 : FloatArray, x2 : FloatArray ) : Float {
        var sum = 0.0f
        for( i in x1.indices ) {
            sum += ( x1[i] - x2[i] ).pow( 2 )
        }
        return sqrt( sum )
    }

    private fun normalizeVector( x : FloatArray ) : FloatArray {
        val mag = sqrt( x.map{ xi -> xi.pow( 2 ) }.sum() )
        return x.map{ xi -> xi/mag }.toFloatArray()
    }

    // Compute the cosine of the angle between x1 and x2.
    private fun cosineSimilarity( x1 : FloatArray , x2 : FloatArray ) : Float {
        var dotProduct = 0.0f
        var mag1 = 0.0f
        var mag2 = 0.0f
        var sum = 0.0f
        for (i in x1.indices) {
            dotProduct += (x1[i] * x2[i])
            mag1 += x1[i].toDouble().pow(2.0).toFloat()
            mag2 += x2[i].toDouble().pow(2.0).toFloat()
            sum += (x1[i] - x2[i]).pow(2)
        }
        mag1 = sqrt(mag1)
        mag2 = sqrt(mag2)
        return dotProduct / (mag1 * mag2)
    }

    private fun BitmaptoNv21( bitmap: Bitmap ): ByteArray {
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

    private fun toBitmap( image : Image ): Bitmap {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, yuvImage.width, yuvImage.height), 50, out)
        val yuv = out.toByteArray()
        return BitmapFactory.decodeByteArray(yuv, 0, yuv.size)
    }

}