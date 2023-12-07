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
import android.util.Log
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.ml.quaterion.facenetdetection.ml.FaceNetModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.util.Collections
import java.util.concurrent.atomic.AtomicInteger

// Utility class to read images from internal storage
class FileReader( private var faceNetModel: FaceNetModel ) {

    private val realTimeOpts = FaceDetectorOptions.Builder()
        .setPerformanceMode( FaceDetectorOptions.PERFORMANCE_MODE_FAST )
        .build()
    private val detector = FaceDetection.getClient( realTimeOpts )

    private val imageData = Collections.synchronizedList( ArrayList<Pair<String,FloatArray>>() )
    private val numImagesWithNoFaces = AtomicInteger( 0 )

    data class FileReaderResult(
        val embeddedFaces: List<Pair<String,FloatArray>> ,
        val numImagesWithNoFaces: Int
    )

    fun run(
        data : ArrayList<Pair<String,Bitmap>> ,
        onResult: ((FileReaderResult) -> Unit)
    ) {
        // Block until all jobs are completed
        runBlocking( Dispatchers.Default ) {
            val mid = data.size / 2 ;
            listOf(
                launch( Dispatchers.Default ) {
                    data.subList( 0 , mid ).forEach {
                        runBlocking( Dispatchers.Default ) {
                            scanImage( it.first, it.second )
                        }
                    }
                } ,
                launch( Dispatchers.Default ) {
                    data.subList( mid + 1 , data.size ).forEach {
                        runBlocking( Dispatchers.Default ) {
                            scanImage( it.first, it.second )
                        }
                    }
                }
            ).joinAll()
            Log.e( "COROUTINES" , "results shown now " + imageData.size )
            onResult( FileReaderResult( imageData , numImagesWithNoFaces.get() ) )
        }
    }


    // Crop faces and produce embeddings ( using FaceNet ) from given image.
    // Store the embedding in imageData
    private fun scanImage(name: String, image: Bitmap ) {
        val inputImage = InputImage.fromBitmap( image , 0 )
        Log.e( "COROUTINES" , "Processing ->  $name" )
        val faces = Tasks.await( detector.process( inputImage ) )
        if ( faces.size != 0 && BitmapUtils.validateRect( image , faces[0].boundingBox ) ) {
            val embedding = faceNetModel.getFaceEmbedding(
                BitmapUtils.cropRectFromBitmap( image, faces[0].boundingBox ) )
            Log.e( "COROUTINES" , "Added to list ->  $name" )
            imageData.add(Pair(name, embedding))
        }
        else {
            Log.e( "COROUTINES" , "No faces -> $name" )
            numImagesWithNoFaces.incrementAndGet()
        }
    }


}