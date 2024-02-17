package com.ml.shubham0204.facenetdetection.ml


class ModelInfo(
    val name : String ,
    val assetsFilename : String ,
    val cosineThreshold : Float ,
    val l2Threshold : Float ,
    val outputDims : Int ,
    val inputDims : Int ,
    val description: String = "" )

class Models {

    companion object {

        val FACENET = ModelInfo(
            "FaceNet" ,
            "facenet.tflite" ,
            0.4f ,
            10f ,
            128 ,
            160
        )

        val FACENET_512 = ModelInfo(
            "FaceNet-512" ,
            "facenet_512.tflite" ,
            0.3f ,
            23.56f ,
            512 ,
            160
        )

        val models = arrayOf(
            FACENET ,
            FACENET_512
        )

    }

}