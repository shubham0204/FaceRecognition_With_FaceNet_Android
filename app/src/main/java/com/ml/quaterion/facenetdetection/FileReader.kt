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
package com.ml.quaterion.facenetdetection

import android.graphics.Bitmap
import android.graphics.Rect
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.ml.quaterion.facenetdetection.model.FaceNetModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// Utility class to read images from internal storage
class FileReader( private var faceNetModel: FaceNetModel ) {

    private val realTimeOpts = FaceDetectorOptions.Builder()
        .setPerformanceMode( FaceDetectorOptions.PERFORMANCE_MODE_FAST )
        .build()
    private val detector = FaceDetection.getClient( realTimeOpts )
    private val defaultScope = CoroutineScope( Dispatchers.Default )
    private val mainScope = CoroutineScope( Dispatchers.Main )
    private var numImagesWithNoFaces = 0
    private var imageCounter = 0
    private var numImages = 0
    private var data = ArrayList<Pair<String,Bitmap>>()
    private lateinit var callback: ProcessCallback

    // imageData will be provided to the MainActivity via ProcessCallback ( see the run() method below ) and finally,
    // used by the FrameAnalyser class.
    private val imageData = ArrayList<Pair<String,FloatArray>>()



    // Given the Bitmaps, extract face embeddings from then and deliver the processed embedding to ProcessCallback.
    fun run( data : ArrayList<Pair<String,Bitmap>> , callback: ProcessCallback ) {
        numImages = data.size
        this.data = data
        this.callback = callback
        scanImage( data[ imageCounter ].first , data[ imageCounter ].second )
    }


    interface ProcessCallback {
        fun onProcessCompleted( data : ArrayList<Pair<String,FloatArray>> , numImagesWithNoFaces : Int )
    }


    // Crop faces and produce embeddings ( using FaceNet ) from given image.
    // Store the embedding in imageData
    private fun scanImage( name : String , image : Bitmap ) {
        mainScope.launch {
            val inputImage = InputImage.fromByteArray(
                BitmapUtils.bitmapToNV21ByteArray(image),
                image.width,
                image.height,
                0,
                InputImage.IMAGE_FORMAT_NV21
            )
            detector.process(inputImage)
                .addOnSuccessListener { faces ->
                    if (faces.size != 0) {
                        mainScope.launch {
                            val embedding = getEmbedding(image, faces[0].boundingBox)
                            imageData.add(Pair(name, embedding))
                            // Embedding stored, now proceed to the next image.
                            if (imageCounter + 1 != numImages) {
                                imageCounter += 1
                                scanImage(data[imageCounter].first, data[imageCounter].second)
                            } else {
                                // Processing done, reset the file reader.
                                callback.onProcessCompleted(imageData, numImagesWithNoFaces)
                                reset()
                            }
                        }
                    }
                    else {
                        // The image contains no faces, proceed to the next one.
                        numImagesWithNoFaces += 1
                        if (imageCounter + 1 != numImages) {
                            imageCounter += 1
                            scanImage(data[imageCounter].first, data[imageCounter].second)
                        } else {
                            callback.onProcessCompleted(imageData, numImagesWithNoFaces)
                            reset()
                        }
                    }
                }
        }
    }

    // Suspend function for running the FaceNet model
    private suspend fun getEmbedding(image: Bitmap, bbox : Rect ) : FloatArray = withContext( Dispatchers.Default ) {
        return@withContext faceNetModel.getFaceEmbedding(
            BitmapUtils.cropRectFromBitmap(
                image,
                bbox
            )
        )
    }


    private fun reset() {
        imageCounter = 0
        numImages = 0
        numImagesWithNoFaces = 0
        data.clear()
    }

}