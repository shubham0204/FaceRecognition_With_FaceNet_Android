package com.ml.shubham0204.facenetdetection.ml

/* This class takes embeddings loaded from the user's file system
* and the one taken from the current camera frame, and compares
* to determine the identity of the person in the current frame
* It calls methods defined in src/main/cpp/src/face_detector.cpp through JNI
*/
class Annotator {

    private var annotatorPtr: Long = 0L
    private var ready: Boolean = false

    companion object {
        init {
            System.loadLibrary( "facedetector" )
        }
    }

    fun initialize(
        scannedEmbeddings: List<Pair<String,FloatArray>>
    ) {
        val names: Array<String> = scannedEmbeddings.map{ it.first }.toTypedArray()
        val embeddings: Array<FloatArray> = scannedEmbeddings.map{ it.second }.toTypedArray()
        this.annotatorPtr = createAnnotator( names , embeddings ,
            128 , 0.4f )
        this.ready = true
    }

    fun run(
        embedding: FloatArray
    ) : String {
        return this.identify( annotatorPtr , embedding )
    }

    private external fun createAnnotator(
        subjectNames: Array<String> ,
        subjectEmbeddings: Array<FloatArray> ,
        embeddingDims: Int ,
        thresholdCosine: Float
    ) : Long

    private external fun identify(
        annotatorPtr: Long ,
        subjectEmbedding: FloatArray
    ) : String

    private external fun releaseAnnotator(
        annotatorPtr: Long
    )

}