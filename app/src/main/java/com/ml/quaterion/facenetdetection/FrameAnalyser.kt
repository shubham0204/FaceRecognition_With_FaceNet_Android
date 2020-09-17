package com.ml.quaterion.facenetdetection

import android.content.Context
import android.graphics.*
import android.media.Image
import android.os.Environment
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.firebase.ml.vision.FirebaseVision
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.common.FirebaseVisionImageMetadata
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions
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
    private val realTimeOpts = FirebaseVisionFaceDetectorOptions.Builder()
        .setPerformanceMode(FirebaseVisionFaceDetectorOptions.FAST)
        .build()
    private val detector = FirebaseVision.getInstance().getVisionFaceDetector(realTimeOpts)

    // Used to determine whether the incoming frame should be dropped or processed.
    private var isProcessing = AtomicBoolean(false)

    // FirebaseImageMeta for defining input image params.
    private var metadata = FirebaseVisionImageMetadata.Builder()
        .setWidth(640)
        .setHeight(480)
        .setFormat(FirebaseVisionImageMetadata.IMAGE_FORMAT_NV21 )
        .setRotation(degreesToFirebaseRotation(90))
        .build()

    // Store the face embeddings in a ( String , FloatArray ) Hashmap.
    // Where String -> name of the person abd FloatArray -> Embedding of the face.
    var faceList = HashMap<String,FloatArray>()

    // Boolean value to check whether the user has enabled auto mode.
    var isAutoMode = false

    // Store newly found faces in ( Name , Embedding ) form.
    private val autoRecognitionFaceList =HashMap<String,FloatArray>()

    // Number of users uniquely identified by the Auto Mode.
    private var autoRecognizedUsersCount : Int = 0

    // Threshold to determine whether the face belongs to a new user or an existing one.
    // If the similarity score is smaller than this threshold, then a new name will be assigned to the user and will
    // be appended in autoRecognitionFaceList
    private val AUTO_RECOGNITION_THRESHOLD = 0.6f

    // FaceNet model utility class
    private val model = FaceNetModel( context )

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

            Log.e( "Infoooooooooooooo " , rotationDegrees.toString() )
            // Perform face detection
            val inputImage = FirebaseVisionImage.fromByteArray( BitmaptoNv21( bitmap ) , metadata )
            detector.detectInImage(inputImage)
                .addOnSuccessListener { faces ->
                    // Start a new thread to avoid frequent lags.
                    Thread {
                        val predictions = ArrayList<Prediction>()
                        for (face in faces) {
                            try {
                                // Crop the frame using face.boundingBox.
                                // Convert the cropped Bitmap to a ByteBuffer.
                                // Finally, feed the ByteBuffer to the FaceNet model.
                                val subject = model.getFaceEmbedding( bitmap , face.boundingBox , true )
                                Log.i( "Model" , "New frame received.")

                                // Auto recognition mode
                                if ( isAutoMode ){

                                    // Determine index and value of the highest similarity score.
                                    var highestSimilarityScore = -1f
                                    var highestSimilarityScoreName = ""
                                    for ( ( name , embedding ) in autoRecognitionFaceList ) {
                                        val p = cosineSimilarity( subject , embedding )
                                        Log.i( "Model" , "Similarity score for ${name} is ${p}.")
                                        if ( p > highestSimilarityScore ) {
                                            highestSimilarityScore = p
                                            highestSimilarityScoreName = name
                                        }
                                    }

                                    // If the highest similarity is smaller than the threshold, we assume that
                                    // this face belongs to a new user.
                                    if ( highestSimilarityScore <= AUTO_RECOGNITION_THRESHOLD ) {
                                        // Update the users' count.
                                        autoRecognizedUsersCount += 1
                                        // Append the user to autoRecognitionFaceList
                                        autoRecognitionFaceList[ "user$autoRecognizedUsersCount" ] = subject
                                        // Show a message on the screen
                                        MainActivity.setMessage( "New User added." )
                                    }
                                    else {
                                        // The face belongs oto an existing user. Show the name and the box on the
                                        // overlay.
                                        predictions.add(
                                            Prediction(
                                                face.boundingBox,
                                                highestSimilarityScoreName
                                            )
                                        )
                                    }
                                }
                                else {
                                    // Determine index and value of the highest similarity score.
                                    var highestSimilarityScore = -1f
                                    var highestSimilarityScoreName = ""
                                    for ( ( name , embedding ) in faceList ) {
                                        val p = cosineSimilarity( subject , embedding )
                                        Log.i( "Model" , "Similarity score for ${name} is ${p}.")
                                        if ( p > highestSimilarityScore ) {
                                            highestSimilarityScore = p
                                            highestSimilarityScoreName = name
                                        }
                                    }
                                    Log.i( "Model" , "Person identified as ${highestSimilarityScoreName} with " +
                                            "confidence of ${highestSimilarityScore * 100} %" )
                                    // Push the results in form of a Prediction.
                                    predictions.add(
                                        Prediction(
                                            face.boundingBox,
                                            highestSimilarityScoreName
                                        )
                                    )
                                }

                            }
                            catch ( e : Exception ) {
                                // If any exception occurs with this box and continue with the next boxes.
                                continue
                            }
                        }

                        // Clear the BoundingBoxOverlay and set the new results ( boxes ) to be displayed.
                        boundingBoxOverlay.faceBoundingBoxes = predictions
                        boundingBoxOverlay.invalidate()

                        // Declare that the processing has been finished and the system is ready for the next frame.
                        isProcessing.set(false)

                    }.start()
                }
                .addOnFailureListener { e ->
                    Log.e("Error", e.message)
                }
        }
    }

    private fun saveBitmap(image: Bitmap, name: String) {
        val fileOutputStream =
            FileOutputStream(File( Environment.getExternalStorageDirectory()!!.absolutePath + "/$name.png"))
        image.compress(Bitmap.CompressFormat.PNG, 100, fileOutputStream)
    }

    // Cosine similarity for two vectors ( face embeddings ).
    // cosineSimilarity = embedding1.dot( embedding2 ) / ||embedding1|| * ||embedding2||
    private fun cosineSimilarity( x1 : FloatArray , x2 : FloatArray ) : Float {
        var dotProduct = 0.0f
        var mag1 = 0.0f
        var mag2 = 0.0f
        for( i in x1.indices ) {
            dotProduct += ( x1[i] * x2[i] )
            mag1 += x1[i].toDouble().pow(2.0).toFloat()
            mag2 += x2[i].toDouble().pow(2.0).toFloat()
        }
        mag1 = sqrt( mag1 )
        mag2 = sqrt( mag2 )
        return dotProduct / ( mag1 * mag2 )
    }


    private fun degreesToFirebaseRotation(degrees: Int): Int = when(degrees) {
        0 -> FirebaseVisionImageMetadata.ROTATION_0
        90 -> FirebaseVisionImageMetadata.ROTATION_90
        180 -> FirebaseVisionImageMetadata.ROTATION_180
        270 -> FirebaseVisionImageMetadata.ROTATION_270
        else -> throw Exception("Rotation must be 0, 90, 180, or 270.")
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