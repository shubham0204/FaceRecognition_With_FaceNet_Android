package com.ml.quaterion.facenetdetection.ml

class Annotator {

    private var annotatorPtr: Long = 0L
    var ready: Boolean = false

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
            128 , 0.4f , 0.0f , "cosine" )
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
        thresholdCosine: Float ,
        thresholdL2: Float ,
        method: String
    ) : Long

    private external fun identify(
        annotatorPtr: Long ,
        subjectEmbedding: FloatArray
    ) : String

    private external fun releaseAnnotator(
        annotatorPtr: Long
    )

}